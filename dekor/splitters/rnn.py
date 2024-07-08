"""
RNN model for splitting German compounds based on the DECOW16 compound data.
"""

import os
import re
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, Iterable, Optional, List, Tuple, Literal, Dict

from dekor.splitters.base import BaseSplitter
from dekor.utils.gecodb_parser import (
    Compound,
    Link,
    parse_gecodb
)
from dekor.utils.vocabs import StringVocab
from dekor.utils.datasets import XYDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RNN(nn.Module):

    def __init__(
            self,
            *,
            vocab_size: int,
            output_size: int,
            hidden_size: Optional[int]=64,
            embedding_dim: Optional[int]=16,
            bidirectional: Optional[bool]=True,
            num_layers: Optional[int]=2,
            dropout: Optional[float]=0.05,
            use_log: Optional[bool]=False,
            batch_size: Optional[int]=4096,
            plot_out_dir: Optional[str]=None,
            plot_step: Optional[int]=250
        ) -> None:

        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.plot_out_dir = plot_out_dir
        self.plot_step = plot_step

        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity="tanh",
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True
        )
        self.linear = nn.Linear(self.D * self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1) if use_log else nn.Softmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        # input: b x 1 (x 1), hidden: D * nl x b x h
        embedded_tensor = self.embedding(input_tensor)  # b x 1 x emd
        output, hidden = self.rnn(embedded_tensor, hidden_tensor)   # b x 1 x D * h, D * nl x b x h
        output = self.linear(output)    # b x 1 x o
        output = output.squeeze(1)  # b x o, for softmax as it works on dimension 1
        output = self.softmax(output)   # b x o
        return output, hidden
    
    @property
    def D(self):
        return 1 if not self.bidirectional else 2

    def init_hidden(self):
        return torch.zeros(self.D * self.num_layers, self.batch_size, self.hidden_size)
    

class RNNSplitter(BaseSplitter):

    name = "rnn"

    def __init__(
            self,
            n: Optional[int]=3,
            eliminate_allomorphy: Optional[bool]=True,
            optimizer: Optional[Literal["sgd", "adamw"]]="adamw",
            criterion: Optional[Literal["crossentropy", "nllloss"]]="crossentropy",
            learning_rate: Optional[float]=0.001,
            n_epochs: Optional[float]=3,
            verbose: Optional[bool]=True,
            **kwargs
        ) -> None:
        assert optimizer in ["sgd", "adamw"]
        assert criterion in ["crossentropy", "nllloss"]
        assert learning_rate > 0 and learning_rate < 1
        assert "use_log" not in kwargs, "Is set automatically"
        self.n = n
        self.eliminate_allomorphy = eliminate_allomorphy
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.vocab_chars = StringVocab()
        self.vocab_links = StringVocab()
        self._elink = Link(
            self.vocab_links.unk,
            span=(-1, -1),
            type=self.vocab_links.unk
        )
        self.model_params = kwargs
        # we want to pass the whole context as an input (since we go character-wise, we can't fit more)
        self.maxlen = self.n * 2 + 3 + 2 # `self.n` from both sides and up to 3-place `mid` and two '|' separators
        self.ignore_index = -100

    def _metadata(self):
        return {
            "n": self.n,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "model": str(self.rnn)
        }

    def pad(self, seq: List[int]) -> List[int]:
        return seq + [self.ignore_index] * (self.maxlen - len(seq)) 

    def _forward(self, compound: Compound) -> List[Tuple[int]]:

        # Analyze a single compound; performed as a sliding window
        # with a sliding window inside
        # over the compound lemma, where for each position it is stored,
        # which left and right context in n-grams there is and what is in between and
        # whether that "in between" is and, if, yes, which one.
        # Example:
        #   "bundestag" with 2-grams
        #   ">b", "un", "" --> no link
        #   ">b", "nd", "u" --> no link
        #   ">b", "de", "un" --> no link
        #   ...
        #   "nd", "es", "" --> no link
        #   "nd", "st", "e" --> no link
        #   "nd", "ta", "es" --> link "_+s_"
        #   ...
        # This triples are then joined to the context and fed to the RNN
        lemma = f'>{compound.lemma}<'    # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0

        # as we know which links to expect, we will track them 
        next_link_idx = 0
        # masks will be of a form (c_l, c_r, c_m, l), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * c_m is the middle n-gram
        #   * l is the link id (unknown id if none)
        position_codes = []
        link_ids = []
        # Make sliding window; however, we want to start not directly with
        # n-grams, but first come from 1-grams to n-grams at the left of the compound
        # and then slide by n-grams; same with the end: not the last n-gram,
        # but n-gram to 1-gram. To be more clear: having 'Bundestag' and 3-grams, we don't want contexts
        # to be directly (("bun", "des"), ("und", "est"), ..., ("des", "tag")), 
        # but we rather want (("b", "und"), ("bu", "nde"), ("bun", "des"), ..., ("des", "tag"), ("est", "ag"), ("sta", "g")).
        # As we process compounds unidirectionally and move left to right,
        # we want subtract max n-gram length to achieve this effect; thus, with a window of length
        # max n-gram length, we will begin with 1-grams, ..., reach n-grams, ..., and end with ..., 1-grams
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            # next expected link; we use empty link in case there are no links anymore to unify the workflow below
            next_link = compound.links[next_link_idx] if next_link_idx < len(compound.links) else self._elink
            s = max(0, i)   # start of left
            m = i + n   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                # define if there is a link incoming at this index;
                if (m - 1, m + r - 1) == next_link.span:    # -1 is correction because of special symbols
                    link = next_link
                    # increment
                    next_link_idx += 1
                else: link = self._elink
                # add
                # link_id = self.vocab_links.add(link.component)
                link_id = self.vocab_links.add(link.component)
                link_ids.append(link_id)
                # join position and encode to fit to RNN
                position = '|'.join([left, right, mid])
                position_code = [
                    self.vocab_chars.add(char)
                    for char in position
                ]
                # pad
                position_code = self.pad(position_code)
                position_codes.append(position_code)

        return position_codes, link_ids

    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        
        hidden = self.rnn.init_hidden()

        # iterate over features
        for i in range(x.size(1)):
            input = x[:, i]
            output, hidden = self.rnn(input, hidden)

        loss = self.criterion(output, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def fit(self, compounds: Iterable[Compound]):

        """
        Feed DECOW16 compounds to the model. That includes iterating through each compound
        with a sliding window and feeding occurrences of links between n-gram contexts
        to the RNN to try to fit to the target distribution.

        Parameters
        ----------
        compounds : `Iterable[Compound]`
            collection of `Compound` objects out of COW dataset to fit

        Returns
        -------
        `NGramsSplitter`
            fit model
        """

        all_position_codes = []
        all_link_ids = []
        progress_bar = tqdm(compounds, desc="Preprocessing") if self.verbose else compounds
        for compound in progress_bar:
            # collect masks from a single compound
            position_codes, link_ids = self._forward(compound)
            all_position_codes += position_codes
            all_link_ids += link_ids
        X = torch.tensor(all_position_codes, dtype=torch.long, device=DEVICE).unsqueeze(-1)
        Y = torch.tensor(all_link_ids, dtype=torch.long, device=DEVICE)

        # we need to replace ignore index; in the RNN, embedding treats inputs as category ids,
        # so it won't recognize -100; since the classes already start with 0, we have to take the next highest one
        new_ignore_index = len(self.vocab_chars)
        X[X == self.ignore_index] = new_ignore_index
        self.ignore_index = new_ignore_index

        # init RNN
        self.rnn = RNN(
            vocab_size=len(self.vocab_chars) + 1,   # plus ignore index
            output_size=len(self.vocab_links),
            **self.model_params,
            use_log=self.criterion == "nllloss"
        )

        self.rnn.train()

        # init optimizer, criterion
        optimizer_class = torch.optim.SGD if self.optimizer == "sgd" else torch.optim.AdamW
        self.optimizer = optimizer_class(self.rnn.parameters(), lr=self.learning_rate)

        criterion_class = nn.CrossEntropyLoss if self.criterion == "crossentropy" else nn.NLLLoss
        self.criterion = criterion_class(ignore_index=self.ignore_index)

        # for plotting
        if self.rnn.plot_out_dir:
            losses = []
            accumulative_loss = 0
            c = self.rnn.plot_step or 1

        # train the RNN
        train_dataloader = DataLoader(
            XYDataset(X, Y),
            batch_size=self.rnn.batch_size,
            drop_last=True
        )
        if self.verbose: progress_bar = tqdm(total=(len(X) * self.n_epochs) // self.rnn.batch_size)
        for i in range(self.n_epochs):
            if self.verbose: progress_bar.set_description_str(f"Epoch {i + 1}")
            for j, (x, y) in enumerate(train_dataloader):
                # accumulative_loss = accumulative_loss + (loss := self._fit_example(x, y))
                if self.verbose: progress_bar.update()
                loss = self._train_on_batch(x, y)
                if self.rnn.plot_out_dir:
                    accumulative_loss += loss / self.rnn.batch_size
                    if j % c == 0:
                        losses.append(accumulative_loss / c)
                        accumulative_loss = 0

        # plot
        if self.rnn.plot_out_dir:
            out_path = os.path.join(self.rnn.plot_out_dir, "rnn.png")
            plt.figure()
            plt.plot(losses)
            plt.xlabel(f"Iterations with step {c}")
            plt.ylabel(f"{self.criterion.__class__.__name__}")
            plt.savefig(out_path, format="png", metadata={"Description": self.__repr__()})

        return self
    
    # def _predict(self, lemma: str, position_codes: List[List[int]], logits: torch.Tensor) -> Compound:
    def _predict(self, lemma: str, pos2log: Dict[str, torch.Tensor]) -> Compound:

        # predict a single lemma and return a DECOW16-format `Compound`
        raw = ""    # will iteratively restore DECOW16-format raw compound
        lemma = f'>{lemma.lower()}<'        # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0
        c = 0   # correction to skip links (see below)

        # imitating same loop to keep track of positions
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            s = max(0, i + c)   # start of left
            m = i + n + c   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            link_candidates = []
            realizations = []
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                position = '|'.join([left, right, mid])
                position_code = [
                    self.vocab_chars.add(char)
                    for char in position
                ]
                # pad
                position_code = self.pad(position_code)
                logits = pos2log['/'.join(map(str, position_code))]
                link_candidates.append(logits)
                # realization
                position_chars = [
                    self.vocab_chars.decode(id) for id in position_code
                    if id != self.ignore_index
                ]
                position = ''.join(position_chars).split('|')
                realization = position[2]   # left, right, -> mid
                realizations.append(realization)
            link_candidates = torch.stack(link_candidates, dim=0)
            # you can consider the whole thing as observations and their values,
            # where realizations are the observations and link ids are values;
            # so `link_candidates` tell us: at row r (= at realization r)
            # there is a distribution D with the most probable id l;
            # argmax returns index in a flattened array, so to get a two-place position, we need to to this
            best_link_ids = (link_candidates == link_candidates.max()).nonzero()
            best_realization_idx, best_link_id = random.choice(best_link_ids).tolist()
            best_realization = realizations[best_realization_idx]

            # top up the raw
            raw += lemma[i - (1 - n) + c] # discard the diff in the loop declaration + add skip step

            # if there is no link, then unknown id is returned
            if best_link_id != self.vocab_links.unk_id:
                best_link = self.vocab_links.decode(best_link_id)
                component, realization, link_type = Compound.get_link_info(
                    best_link,
                    eliminate_allomorphy=self.eliminate_allomorphy
                )
                # heuristically filter out predictions that cannot be correct (e.g. addition link)
                # in position there can be no position link;
                # also don't forget that when eliminating allomorphy, realizations might not 
                # match but theoretical realization is just empirical realization with schwa prefix
                non_empty_deletion = (link_type == "deletion" and len(best_realization) > 0)
                non_empty_concatenation = (link_type == "concatenation" and len(best_realization) > 0)
                ill_addition = ("addition" in link_type and not re.match(f"^e?{realization}$", best_realization))
                if non_empty_deletion or non_empty_concatenation or ill_addition:
                    continue
                if link_type == "addition_umlaut":
                    raw = Compound.reverse_umlaut(raw)
                elif link_type == "deletion":
                    to_delete = Compound.get_deletion(component)
                    raw += to_delete
                # NOTE: reconstruct from components?
                # If allomorphy is eliminated, we can predict cases like
                # tag_+s_ticket correctly and know that the realization is -es-;
                # however, since we dont reconstruct the compound from components,
                # if we pass tag_+s_ticket to `Compound`, it will think that the
                # realization is -s- even though we know it is not the case.
                # That is why, if eliminated allomorphy encountered,
                # we must reconstruct the link as if allomorphy does not get eliminated,
                # and then `Compound` will still parse the link with elimination
                # but will receive the correct realization we predicted.
                if best_realization != realization:
                    component = re.sub(realization, best_realization, component)
                raw += component

                # When we encounter a link, we know for sure that there can not
                # be another link after it (at least in v4 implementation).
                # That is why we want to skip the link after we found it.
                # For example, if we have "bundestag" and the model decided that after "nd",
                # there is an "es", there is no sense for us to add "es" and
                # start further with "de"; we want to continue straight to "ta".
                # However, we cannot just assign a higher `i` because it
                # will reset to its anticipated value in the new iteration,
                # so we have to maintain a correction to add to `i`
                # in order to be sure we are skipping the link. 
                c += len(best_realization)

        raw += lemma[-2] # complete raw when the window has sled, -1 for EOS

        pred = Compound(raw, eliminate_allomorphy=self.eliminate_allomorphy)

        return pred
    
    def _get_positions(self, lemma: str) -> List[List[int]]:

        lemma = f'>{lemma.lower()}<'        # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0

        position_codes = []
        # iterate over lemma to obtain positions
        # same sliding window
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            s = max(0, i)   # start of left
            m = i + n   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                # will return unknown id if unknown
                position = '|'.join([left, right, mid])
                position_code = [
                    self.vocab_chars.add(char)
                    for char in position
                ]
                # pad
                position_code = self.pad(position_code)
                position_codes.append(position_code)
        
        return position_codes

    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        
        hidden = self.rnn.init_hidden()

        # iterate over features
        for i in range(x.size(1)):
            input = x[:, i]
            output, hidden = self.rnn(input, hidden)

            return output
    
    def predict(self, compounds: List[str]) -> List[Compound]:

        self.rnn.eval()

        all_position_codes = []
        # keep track of milestones to know which codes belong to which compound
        milestones = [0]
        progress_bar = tqdm(compounds, desc="Preprocessing") if self.verbose else compounds
        for compound in progress_bar:
            # collect masks from a single compound
            position_codes = self._get_positions(compound)
            milestones.append(milestones[-1] + len(position_codes))
            all_position_codes += position_codes
        X = torch.tensor(all_position_codes, dtype=torch.long, device=DEVICE).unsqueeze(-1)

        all_logits = []
        test_dataloader = DataLoader(
            XYDataset(X),
            batch_size=self.rnn.batch_size,
            drop_last=False
        )
        if self.verbose: progress_bar = tqdm(total=len(X) // self.rnn.batch_size, desc="Predicting")
        for x in test_dataloader:
            if self.verbose: progress_bar.update()
            if len(x) < self.rnn.batch_size:    # last batch
                # in this case, we want to pad the whole batch to normal size
                # and then drop excessive predictions
                diff = self.rnn.batch_size - len(x)
                pads = [self.pad([])] * diff
                pads = torch.tensor(pads, dtype=torch.long, device=DEVICE).unsqueeze(-1)
                x = torch.cat((x, pads), dim=0)
                logits = self._predict_batch(x)
                logits = logits[:-diff]
            else:
                logits = self._predict_batch(x)    
            # since the mapping between positions and link ids is not straightforward
            # (there are 4 mid size at one position and we should consider them at once),
            # we cannot just get the max probability ids here, we will have to
            # pass them to `_predict()` to unscramble there
            all_logits += logits

        all_logits = torch.stack(all_logits, dim=0)

        # pos2log = {
        #     '/'.join(position_codes + [i]): logits
        #     for i, (position_codes, logits) in enumerate(zip(all_position_codes, all_logits))
        # }
        
        preds = []
        progress_bar = tqdm(compounds, desc="Postprocessing") if self.verbose else compounds
        for i, compound in enumerate(progress_bar):
            start, end = milestones[i], milestones[i + 1]
            position_codes = all_position_codes[start: end]
            logits = all_logits[start: end]
            pos2log = {
                '/'.join(map(str, position_code)): logit  # there is about 0.02-0.03% words with collisions
                for position_code, logit in zip(position_codes, logits)
            }
            pred = self._predict(compound, pos2log)
            preds.append(pred)

        return preds


if __name__ == "__main__":

    gecodb = parse_gecodb('./resources/gecodb_v04.tsv', eliminate_allomorphy=True, min_count=10000)

    train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
    train_compounds = train_data["compound"].values
    test_compounds = test_data["compound"].values
    test_lemmas = [
        compound.lemma for compound in test_compounds
    ]

    splitter = RNNSplitter(
        n_epochs=1,
        n=2,
        plot_out_dir="./model_runs/run_info/rnn",
        batch_size=4096
    ).fit(train_compounds)

    pred_compounds = np.array(splitter.predict(test_lemmas))