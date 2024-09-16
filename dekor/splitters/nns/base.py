"""
Base NN-based model for splitting German compounds based on the DECOW16 compound data.
"""

import re
import random
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from io import BytesIO
from tqdm import tqdm
from typing import Optional, Iterable, Optional, List, Self, Literal, Dict

from dekor.splitters.base import BaseSplitter, DEVICE
from dekor.utils.gecodb_parser import (
    Compound,
    UMLAUTS_REVERSED
)
from dekor.utils.vocabs import StringVocab
from dekor.utils.datasets import XYDataset


class BaseNN(ABC, nn.Module):

    """
    Backbone network for NN-based splitters.
    """

    def __init__(self, **kwargs) -> None:
        super(BaseNN, self).__init__()
        for param, value in kwargs.items():
            setattr(self, param, value)
        self._build_self()   

    @abstractmethod
    def _build_self(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, *args, **kwargs) -> int:
        pass


class BaseRecurrentNN(BaseNN):  # RNN, GRU

    def forward(self, input_tensor, hidden_tensor, force_softmax=False):
        # input: b x 1 (x 1), hidden: D * nl x b x h
        embedded_tensor = self.embedding(input_tensor)  # b x 1 x emd
        output, hidden = self.recurrent(embedded_tensor, hidden_tensor)   # b x 1 x D * h, D * nl x b x h
        output = self.linear(output)    # b x 1 x o
        output = output.squeeze(1)  # b x o, for softmax as it works on dimension 1
        if self.require_softmax or force_softmax: output = self.softmax(output)   # b x o
        return output, hidden
    
    @property
    def D(self):
        return 1 if not self.bidirectional else 2

    def init_hidden(self):
        return torch.zeros(self.D * self.num_layers, self.batch_size, self.hidden_size)


class BaseNNSplitter(BaseSplitter):

    """
    Base class for NN-based splitters.

    Parameters
    ----------

    n : `int`, optional, defaults to `3`
        length of the contexts to encode on the left and on the right from
        target position (which is either a link or significant absence of it)
        for fitting and prediction

    record_none_links : `bool`, optional, defaults to `False`
        whether to record contexts between which no links occur;
        hint: that could lead to a strong bias towards no link choice

    eliminate_allomorphy : `bool`, optional, defaults to `True`
        whether to eliminate allomorphy of the input link, e.g. _+es_ to _+s_

    optimizer : `str`, one of ["sgd", "adamw"], optional, defaults to "adamw"
        name of the optimizer for the backbone NN training; `torch.nn.SGD` or `torch.nn.AdamW`
        are used respectively

    criterion : `str`, one of ["crossentropy", "bce", "margin"], optional, defaults to "crossentropy"
        name of the loss function for the backbone NN training; `torch.nn.CrssEntropyLoss`,
        `torch.nn.BCEWithLogitsLoss`, or `torch.nn.MultiLabelSoftMarginLoss` are used respectively

    learning_rate : `float`, optional, defaults to `0.001`
        learning rate for the backbone NN training; must be in an interval (0; 1)

    n_epochs : `int`, optional, defaults to `3`
        number of epochs for the backbone NN to train; must be positive

    save_plot : `bool`, optional, defaults to `False`
        whether to save training plot with losses; if `True`, binary representation
        of the plot in PNG will be stored in the `plot_buffer` attribute

    verbose : `bool`, optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds

    kwargs:
        parameters to pass to the backbone NN
    """

    # When doing benchmarking, it is computationally inefficient to combine all the 
    # parameters that we can pass both to the wrapper and the backbone NN 
    # (too much configurations). Correspondingly, we should separate the parameters
    # into groups and run benchmarking inside those groups separately, "freezing"
    # parameters of other groups; thus, we can reassemble intuitively the best parameters
    # group by group.
    # We decided to divide the parameters by their "functionality". The final order is:
    #   1. hyperparameters of the backbone NN;
    #   2. parameters of NN training;
    #   3. wrapper parameters, i.e. parameters of feature retrieval and managements.
    # In ordering the parameters, we were guided by the following logic:
    #   * Whatever the wrapper parameters are, the extracted features will still bear
    #   information about contexts vs links distribution; that is, whatever wrapper parameters
    #   we set, the "competition" of different configurations from the two remaining groups will be fair
    #   because they will compete over the same information whatever this information is.
    #   In other words, if one configuration of parameters wins another on one set of features,
    #   it will probably win over another features, because both these features
    #   describe the same distribution (just with different quality).
    #   So this group should be tested the last.
    #   * From the 2 remaining groups, hyperparameters are the core of the whole model;
    #   if the hyperparameters are chosen poorly, the performance will be bad
    #   no matter which training parameters are picked. However, even with a poor choice of
    #   training parameters the model can capture the patterns.
    #   It therefore makes sense to first test different hyperparameters, because different hyperparameters
    #   will show different performance even with badly matching training parameters and, hence,
    #   will be able to be ranked, but different training parameters will all result in 
    #   low performance if the hyperparameters are chosen erroneously.

    def __init__(
            self,
            *,
            n: Optional[int]=3,
            record_none_links: bool,
            eliminate_allomorphy: bool,
            optimizer: Optional[Literal["sgd", "adamw"]]="adamw",
            criterion: Optional[Literal["crossentropy", "bce", "margin"]]="crossentropy",
            learning_rate: Optional[float]=0.001,
            n_epochs: Optional[int]=3,
            save_plot: Optional[bool]=False,
            verbose: Optional[bool]=True,
            **model_params
        ) -> None:
        assert optimizer in ["sgd", "adamw"]
        assert criterion in ["crossentropy", "bce", "margin"]
        assert learning_rate > 0 and learning_rate < 1
        assert n_epochs > 0
        assert "require_softmax" not in model_params, "Is set automatically"
        self.n = n
        self.record_none_links = record_none_links
        self.eliminate_allomorphy = eliminate_allomorphy
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.plot_buffer = BytesIO() if save_plot else None
        self.verbose = verbose
        self.vocab_chars = StringVocab()
        self.vocab_links = StringVocab()
        self.model_params = model_params
        # we want to pass the whole context as an input (since we go character-wise, we can't fit more)
        self.maxlen = self.n * 2 + 3 + 2 # `self.n` from both sides and up to 3-place `mid` and two '|' separators
        self.ignore_index = -100

    def _metadata(self) -> dict:
        return {
            "n": self.n,
            "record_none_links": self.record_none_links,
            "eliminate_allomorphy": self.eliminate_allomorphy,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "model": str(self.model)
        }

    def pad(self, seq: List[int]) -> List[int]:
        return seq + [self.ignore_index] * (self.maxlen - len(seq))
    
    # The thing is all the NN-based models work according to the same scenario,
    # that is why this base model mostly implements all the needed methods.
    # It works as follows:
    #   1 `fit()`. Fit compounds:
    #       1.1 `_forward()`. For each compound, <left, right, mid> contexts are collected iteratively,
    #       and they get encoded character-wise; for each context, the link is stored;
    #       1.2. Character codes become X, encoded links - Y;
    #       1.3 `_build_nn()`. The backbone NN is built;
    #       1.4. `_train_on_batch()`. The backbone NN trains batch-wise on X, Y
    #   2. `predict()`. Predict lemmas:
    #       2.1 `_forward()`. For each lemma, <left, right, mid> contexts are collected iteratively,
    #       and they get encoded character-wise; no links are stored;
    #       2.2. Character codes become X, encoded links - Y;
    #       2.3 `_predict_batch()`. The backbone NN predicts Y on X and yields generations
    #       for all positions for the whole batch
    #       2.4. `_predict()`. For each lemma, only its predictions from the batch are taken
    #       (for all its positions), and the prediction is iteratively unscrambled.
    #
    # The fact is, all of those processes are absolutely identical for the NN-based models,
    # so we decided to implement them in one place so here, all algorithmic procedures are covered.
    # Even the training goes the same way: gather X, Y and train batch-wise (but batch training differs).
    # What differentiates the models is the backbone NNs, and so such this base model
    # allows to concentrate only on their aspects in their respective classes, such as
    # `_train_on_batch()`, `_predict_batch()`.
    
    def _forward(
        self,
        compound: Compound,
        add_new: Optional[bool]=True    # while reuseing the function in prediction, we should switch off adding new entries
    ) -> List[List[int]]:

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
        # This triples are then encoded character-wise and fed to the backbone NN
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
        # define whether we should add new characters (yes when training, no when predicting)
        encode_func = self.vocab_chars.add if add_new else self.vocab_chars.encode
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
                else:
                    if self.record_none_links: link = self._elink
                    else: continue
                # add
                # link_id = self.vocab_links.add(link.component)
                link_id = self.vocab_links.add(link.component)
                link_ids.append(link_id)
                # join position and encode to fit to backbone NN
                position = '|'.join([left, right, mid])
                position_code = [
                    encode_func(char)
                    for char in position
                ]
                # pad to the maximum length
                position_code = self.pad(position_code)
                position_codes.append(position_code)

        return position_codes, link_ids
    
    @abstractmethod
    def _build_nn(self) -> None:
        pass
    
    @abstractmethod
    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        pass

    def fit(self, compounds: Iterable[Compound]) -> Self:

        """
        Feed DECOW16 compounds to the model. That includes iterating through each compound
        with a sliding window, collecting and encoding occurrences of links between n-gram contexts
        and training the backbone NN on them to try to fit to the target distribution.

        Parameters
        ----------
        compounds : `Iterable[Compound]`
            collection of `Compound` objects out of COW dataset to fit

        Returns
        -------
        A subclass of `BaseNNSplitter`
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
        X = torch.tensor(all_position_codes, dtype=torch.long, device=DEVICE)
        Y = torch.zeros((len(all_link_ids), len(self.vocab_links)), dtype=torch.long, device=DEVICE)
        # make target the same shape as the input for different losses; treat
        # each row as a holistic distribution with only correct link probability equal to 1;
        # that is needed for the BCE and margin losses
        Y[range(len(all_link_ids)), all_link_ids] = 1

        # we need to replace ignore index sometimes; e.g. in the backbone NN, embedding treats inputs as category ids,
        # so it won't recognize -100; since the classes already start with 0, we have to take the next highest one
        new_ignore_index = len(self.vocab_chars)
        X[X == self.ignore_index] = new_ignore_index
        self.ignore_index = new_ignore_index

        # init model, adds .model attribute
        self._build_nn()

        # train mode
        self.model.train()

        # init optimizer, criterion
        optimizer_class = torch.optim.SGD if self.optimizer == "sgd" else torch.optim.AdamW
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)

        # pass weights to handle disbalance
        class_weights = [self.vocab_links.counts[id] for id in self.vocab_links._vocab_reversed]
        class_weights = torch.tensor(class_weights)
        # normalize
        class_weights = class_weights / class_weights.sum()
        # higher weights for rarer classes
        class_weights = 1 - class_weights
        criterion_class = (
            nn.CrossEntropyLoss if self.criterion == "crossentropy"
            else nn.BCEWithLogitsLoss if self.criterion == "bce"
            else nn.MultiLabelSoftMarginLoss
        )
        # ignore index not supported with Y being a distribution
        self.criterion = criterion_class(weight=class_weights)

        # for plotting
        if self.plot_buffer:
            losses = []
            accumulative_loss = 0
            c = 250

        # train the backbone NN
        train_dataloader = DataLoader(
            XYDataset(X, Y),    # will output batches of x's and y's
            batch_size=self.model.batch_size,
            drop_last=True
        )
        if self.verbose: progress_bar = tqdm(total=(len(train_dataloader) * self.n_epochs))
        for i in range(self.n_epochs):
            if self.verbose: progress_bar.set_description_str(f"Epoch {i + 1} / {self.n_epochs}")
            for j, (x, y) in enumerate(train_dataloader):
                # accumulative_loss = accumulative_loss + (loss := self._fit_example(x, y))
                if self.verbose: progress_bar.update()
                loss = self._train_on_batch(x, y)
                if self.plot_buffer:
                    accumulative_loss += loss / self.model.batch_size
                    if j % c == 0:
                        losses.append(accumulative_loss / c)
                        accumulative_loss = 0

        # plot
        if self.plot_buffer:
            plt.figure()
            plt.plot(losses)
            plt.xlabel(f"Iterations with step {c}")
            plt.ylabel(f"{self.criterion.__class__.__name__}")
            # the plot is saved to the plot buffer and not to a file;
            # if you need the plot, you can easily get it from the buffer, e.g.
            # ```python
            # from PIL import Image
            # from PIL.PngImagePlugin import PngInfo

            # self.plot_buffer.seek(0)
            # plot = Image.open(self.plot_buffer)
            # info = PngInfo()
            # for key, value in plot.text.items(): info.add_text(key, value)
            # plot.save(path, format="png", pnginfo=info)
            # ``` 
            plt.savefig(self.plot_buffer, format="png", metadata={"Description": self.__repr__()})
            plt.close()

        return self
    
    def _predict(self, lemma: str, pos2log: Dict[str, torch.Tensor]) -> Compound:

        # unscramble a single lemma and return a DECOW16-format `Compound`;
        # see explanation in `predict()`
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
                    self.vocab_chars.encode(char)   # no adding new chars while predicting
                    for char in position
                ]
                # pad
                position_code = self.pad(position_code)
                # lookup logits by position
                logits = pos2log['/'.join(map(str, position_code))]
                link_candidates.append(logits)
                # realization
                realization = position.split('|')[2]   # left, right, -> mid
                realizations.append(realization)
            link_candidates = torch.stack(link_candidates, dim=0)

            # top up the raw
            raw += lemma[i - (1 - n) + c] # discard the diff in the loop declaration + add skip step

            # There is a set of heuristics that have to be filtered out in place
            # in order to get cleaner result; for example, there can not be an umlaut link
            # in case there is no umlaut before it. This can mostly be checked once a link with
            # its representation is parsed; however, it is highly inefficient to do that
            # will all links whose probability is more that 0. Instead, we will treat the
            # probabilities as a stack with probs ordered from highest to lowest;
            # thus, at each iteration, we will check if the current max probable link
            # passes the filter and if yes, we'll break the cycle, if no, zero this prob
            # and take the next highest probable one. Thus, at the end we will output
            # the most probable link of all that passed the filter.  
            while True:

                # it cannot exceed because even if all non-zero links will appear invalid,
                # the zero link will at some point become the most probable one and will break the loop

                # you can consider the whole thing as observations and their values,
                # where realizations are the observations and link ids are values;
                # so `link_candidates` tell us: at row r (= at realization r)
                # there is a distribution D with the most probable id l;
                # argmax returns index in a flattened array, so to get a two-place position, we need to to this
                best_link_ids = (link_candidates == link_candidates.max()).nonzero()
                best_realization_idx, best_link_id = random.choice(best_link_ids).tolist()  # `tolist()` to "untenzorize"
                best_realization = realizations[best_realization_idx]

                # if there is no link, then unknown id is returned
                if best_link_id != self.vocab_links.unk_id:

                    best_link = self.vocab_links.decode(best_link_id)
                    component, realization, link_type = Compound.get_link_info(
                        best_link,
                        eliminate_allomorphy=self.eliminate_allomorphy
                    )

                    # heuristically filter out predictions that cannot be correct
                    if (    # use if so that no further checks are performed once one has failed
                        # deletion type with addition realization
                        (link_type == "deletion" and len(best_realization) > 0) or
                        # concatenation type with addition realization
                        (link_type == "concatenation" and len(best_realization) > 0) or
                        # impossible addition; there might be or not be an e-, depends on whether we eliminate allomorphy
                        ("addition" in link_type and not re.match(f"^e?{realization}$", best_realization)) or
                        # umlaut link where there is no umlaut before; test only last stem, i.e. part after the last "_"
                        ("umlaut" in link_type and not re.search('|'.join(UMLAUTS_REVERSED.keys()), raw.split("_")[-1]))
                    ):
                        # zero this impossible link prob
                        link_candidates[best_realization_idx, best_link_id] = 0
                        continue

                    # unfuze raw
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

                    break

                else: break

        raw += lemma[-2] # complete raw when the window has sled, -1 for EOS

        pred = Compound(raw, eliminate_allomorphy=self.eliminate_allomorphy)

        return pred
    
    def _pad_batch(self, batch: torch.Tensor) -> torch.Tensor:
        diff = self.model.batch_size - len(batch)
        pads = [self.pad([])] * diff
        # we need integers, but that will be done in `_predict_batch()`
        pads = torch.tensor(pads, dtype=torch.long, device=DEVICE)
        padded_batch = torch.cat((batch, pads), dim=0)
        return padded_batch

    @abstractmethod
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, lemmas: List[str]) -> List[Compound]:

        """
        Make prediction from lemmas to DECOW16-format `Compound`s

        Parameters
        ----------
        lemmas : `List[str]`
            lemmas to predict

        Returns
        -------
        `List[Compound]`
            preds in DECOW16 compound format
        """

        # Unlike N-gram model, we cannot predict every lemma separately.
        # To be more precise, we can, but that would just be very inefficient. 
        # Instead, we will predict batch-wise; now, the tricky thing here is that
        # batches do not correspond to the bounds or the number of the input lemmas
        # anyhow; because each lemma is of different lengths, it will yield an 
        # arbitrary number of positions; for example a batch of 16 can have
        # 3 last positions of lemma1, then all 7 of lemma2, all 4 of lemma3 and the last one of lemma5.
        # For that reason, when collecting the positions of input lemmas, we will
        # store the boundaries of the masks belonging to them:
        # 0-4 is the 4 positions of lemma1, 4-7 is from lemma2, ...
        # Thus, when we obtain all the probabilities from the backbone NN
        # (we will concatenate all batches), we will no for sure which probs
        # correspond to which lemma. Then, we will have to unscramble
        # the final prediction by iterating over the given lemma
        # with the probs on each step.
        # Just to make it straightforward: what is done in-place in the N-gram model
        # (get position and predict the most probable link), is done outside
        # in NN-based models because it is executed not for a single lemma,
        # but for the whole batch; these results are then interpreted to get the final prediction.

        # to gather positions where there are no links, we force set `record_none_links`
        # to be able to reutilize `_forward()` (there are no positions otherwise);
        # it will not affect training as it's already passed as well as predictions
        # because prediction depends on the weights model learned
        record_none_links_orig = bool(self.record_none_links)   # copy
        self.record_none_links = True

        # eval mode
        self.model.eval()

        all_position_codes = []
        # keep track of milestones to know which codes belong to which lemma
        milestones = [0]
        progress_bar = tqdm(lemmas, desc="Preprocessing") if self.verbose else lemmas
        for lemma in progress_bar:
            # collect masks from a single compound
            dummy_compound = Compound(lemma) # since there are no link in lemma, a single stem will be there
            position_codes, _ = self._forward(dummy_compound, add_new=False)   # no link ids are needed (they're unks anyways)
            milestones.append(milestones[-1] + len(position_codes))
            all_position_codes += position_codes
        X = torch.tensor(all_position_codes, dtype=torch.long, device=DEVICE)

        # return `record_none_links_orig`
        self.record_none_links = record_none_links_orig

        all_logits = [] # we interchange probs and logits in this context
        test_dataloader = DataLoader(
            XYDataset(X),   # will only output batches of x's
            batch_size=self.model.batch_size,
            drop_last=False # we cannot drop last because then not all the lemmas will be predicted
        )
        if self.verbose: progress_bar = tqdm(total=len(test_dataloader), desc="Predicting")
        for x in test_dataloader:
            if self.verbose: progress_bar.update()
            if len(x) < self.model.batch_size:    # last batch
                # in this case, we want to pad the whole batch to normal size
                # and then drop excessive predictions
                diff = self.model.batch_size - len(x)
                x = self._pad_batch(x)
                logits = self._predict_batch(x)
                logits = logits[:-diff]
            else:
                logits = self._predict_batch(x)    
            all_logits += logits

        all_logits = torch.stack(all_logits, dim=0)
        
        preds = []
        progress_bar = tqdm(lemmas, desc="Postprocessing") if self.verbose else lemmas
        for i, lemma in enumerate(progress_bar):
            start, end = milestones[i], milestones[i + 1]
            # get positions and logits for the lemma
            position_codes = all_position_codes[start: end]
            logits = all_logits[start: end]
            # It is quite complicated to try and define the relevant logits
            # just by the current position index. When a link is predicted,
            # some characters might get skipped and it it would be really difficult
            # to keep the track of which logits to skip, because depending
            # on the proximity to the beginning/end of the lemma, different
            # number of logits can belong to one left context index.
            # Considering that "vanilla way" would complicate the loop too much.
            # As alternative, we suggest to define a mapping between <left, right, mid>
            # and it's respective distribution of link probabilities; thus, we will not
            # have to care about how many logits to skip, only characters (which we can).
            # There might be occasional collisions, but they are rare and can be overlooked.
            pos2log = {
                '/'.join(map(str, position_code)): logit  # there is about 0.02-0.03% words with collisions
                for position_code, logit in zip(position_codes, logits)
            }
            pred = self._predict(lemma, pos2log)
            preds.append(pred)

        return preds
    

class BaseForwardNNSplitter(BaseNNSplitter):

    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        output = self.model(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # force softmax is used with CrossEntropyLoss because the loss function
            # doesn't require softmax so it's omitted but it is required during
            # prediction to get probability distribution
            output = self.model(x, force_softmax=True)
            output = output.detach()
        return output
    

class BaseRecurrentNNSplitter(BaseNNSplitter):

    def _train_on_batch(self, x: torch.Tensor, y: torch.Tensor) -> int:
        hidden = self.model.init_hidden()
        # iterate over features
        for i in range(x.size(1)):
            # for embeddings, they must be wrapped into 1 x 1 tensors
            input = x[:, i].unsqueeze(-1)
            output, hidden = self.model(input, hidden)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.model.init_hidden()
            # iterate over features
            for i in range(x.size(1)):
                input = x[:, i].unsqueeze(-1)
                # force softmax is used with CrossEntropyLoss because the loss function
                # doesn't require softmax so it's omitted but it is required during
                # prediction to get probability distribution
                output, hidden = self.model(input, hidden, force_softmax=True)
            output = output.detach()
        return output