"""
N-gram model for splitting German compounds based on the DECOW16 compound data.
"""

import os
import json
import time
from itertools import product
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Iterable, Optional, List, Tuple, Union

from dekor.gecodb_parser import (
    Compound,
    Link,
    parse_gecodb
)
from dekor.eval.evaluate import CompoundEvaluator


class StringVocab:
    
    """
    Simple vocabulary for string occurrences.
    """

    def __init__(self) -> None:
        self._vocab = {self.unk: self.unk_id}
        self._vocab_reversed = {self.unk_id: self.unk}

    @property
    def unk(self):
        return "<unk>"

    @property
    def unk_id(self):
        return 0

    def add(self, seq: Union[str, Iterable[str]]) -> int:
        
        """
        Adds a sequence to vocabulary and assigns it a unique id.
        The sequence is either a single string or an iterable of those.

        Parameters
        ----------
        seq : `Union[str, Iterable[str]]`
            Sequence to add

        Returns
        -------
        `int`
            either the assigned to the newly added sequence id
            in case the added string was not existent in the vocabulary,
            or the id of the sequence in case it already existed
        
        Note
        ----
        Not a string but an iterable of those is given, 
        the sequence will be concatenated using a "!:!" separator and transformed into a single string
        in the vocabulary. However, this is an internal process and you don't have 
        to do that manually to decode / encode a sequence. You can do that
        with the usual `encode()` and `decode()` methods. 
        """

        if isinstance(seq, str): seq = [seq]
        string = '!:!'.join(seq)
        if not string in self._vocab:
            id = len(self)
            self._vocab[string] = id
            self._vocab_reversed[id] = string
        else:
            id = self.encode(string)
        return id

    def encode(self, seq: Union[str, Iterable[str]]) -> int:

        """
        Get the code of a vocabulary entry.
        The entry is either a single string or an iterable of those.

        Parameters
        ----------
        seq : `Union[str, Iterable[str]]`
            target entry

        Returns
        -------
        `int`
            id of the target entry; returns unknown id
            in case the entry is not present in the vocabulary 
        """

        if isinstance(seq, str): seq = [seq]
        string = '!:!'.join(map(str, seq))
        return self._vocab.get(string, self.unk_id)
    
    def decode(self, id: int) -> Union[str, Tuple[str]]:

        """
        Get the vocabulary entry by its code. If entry is an encoded an iterable of those,
        method will be split it and return as `Tuple[str]`, otherwise `str`

        Parameters
        ----------
        id : `id`
            target id

        Returns
        -------
        `Union[str, Tuple[str]]`
            entry of the vocabulary encoded under the target id; returns unknown token
            in case the id is unknown to the vocabulary 
        """

        seq = self._vocab_reversed.get(id, self.unk)
        if '!:!' in seq: seq = seq.split('!:!')
        return seq
    
    def __len__(self) -> int:
        return len(self._vocab)


class NGramsSplitter:

    """
    N-grams-based compound splitter that relies on the COW dataset format.
    First, fits train COW entries, then predicts lemma splits in this format.

    Parameters
    ----------
    n : `int`, optional, defaults to `2`
        maximum n-grams length

    record_none_links : `bool`, optional, defaults to `False`
        whether to record contexts between which no links occur;
        hint: that could lead to a strong bias towards no link choice

    verbose : `bool`, optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds
    """

    def __init__(
            self,
            n: Optional[int]=2,
            record_none_links: Optional[bool]=False,
            verbose: Optional[bool]=True
        ) -> None:
        self.n = n
        self.record_none_links = record_none_links
        self.verbose = verbose
        self.vocab_links = StringVocab()
        self.vocab_positions = StringVocab()
        self._elink = Link(
            self.vocab_links.unk,
            span=(-1, -1),
            type=self.vocab_links.unk
        )

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
        # Later, uniques context-link triples will be counted
        # and this counts information will be used in prediction.
        lemma = f'>{compound.lemma}<'    # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0

        # as we know which links to expect, we will track them 
        next_link_idx = 0
        # masks will be of a form (c_l, c_r, c_m, l), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * c_m is the middle n-gram
        #   * l is the link ("<unk>" if none)
        masks = []
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
                    if self.record_none_links:
                        link = self._elink
                    else: continue
                # add the mask
                link_id = self.vocab_links.add(link.component)
                masks.append((left, right, mid, link_id))

        return masks

    def fit(self, compounds: Iterable[Compound]):

        """
        Feed COW compounds to the model. That includes iterating through each compound
        with a sliding window and counting occurrences of links between n-gram contexts
        to try to fit to the target distribution.

        Parameters
        ----------
        compounds : `Iterable[Compound]`
            collection of `Compound` objects out of COW dataset to fit

        Returns
        -------
        `NGramsSplitter`
            fit model
        """

        if self.verbose:
            self.progress_bar = tqdm(
                total=len(compounds),
                # leave=True,
                desc="Fitting compounds"
            )

        all_masks = []
        for compound in compounds:
            # collect masks from a single compound
            masks = self._forward(compound)
            all_masks += masks
            if self.verbose: self.progress_bar.update()
        all_masks = np.array(all_masks)

        # at this point, all the vocabs are populated, so we can process masks
        # 1. process link counts
        # a mask has a form (c_l, c_r, c_m, l), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * c_m is the mid n-gram
        #   * l is the link ("<unk>" if none)
        # We want to get a mapping from <c_l, c_r, c_m> of contexts C = (c1, c2, ...)
        # to distribution of links L = (l1, l2, ...).
        # If we do that with matrices, they become too large to operate over because they become extremely sparse
        # and take up the whole memory (they are also 4D matrices which makes it worse);
        # that is why we need a more compact way to encode positions.
        # What we can do is to encode positions <c_l, c_r, c_m> separately as strings and thus reduce
        # the 4D matrix of shape (|C|, |C|, |C|, |L|) to a 2D matrix of shape (|P|, |L|),
        # where P is the set of encoded positions. Moreover, we will this significantly
        # reduce the sparseness of the matrix because there will be no unknowns contexts
        # so no zeros between n-grams, only zeros on L distribution when a link never occurs in a position.
        _, unique_indices, counts = np.unique(all_masks, return_index=True, return_counts=True, axis=0)
        # `np.unique()` sorts masks so the original order is not preserved;
        # to recreate it, the method returns indices of masks in the original dataset
        # in order of their counts; for example, `unique_indices[5]` contains
        # the index of that mask in `masks_links` whose count is under `counts[5]`
        unique_masks_links = all_masks[unique_indices]
        positions = unique_masks_links[:, :3]
        # add_to_vocab = lambda seq: self.vocab_positions.add(seq)
        positions_ids = np.apply_along_axis(
            self.vocab_positions.add,
            arr=positions,
            axis=1
        )
        link_ids = unique_masks_links[:, 3].astype(np.int32)
        # heuristically force <unk> link from <unk> context;
        # add them here because sparse arrays don't allow item assignment
        positions_ids = np.append(positions_ids, self.vocab_positions.unk_id)
        link_ids = np.append(link_ids, self.vocab_links.unk_id)
        counts = np.append(counts, 1)

        n_pos, n_links = len(self.vocab_positions), len(self.vocab_links)
        counts_links = np.zeros((n_pos, n_links), dtype=np.float32)  #f or further division
        counts_links[positions_ids, link_ids] = counts

        # counts to frequencies; since all positions point at least to <unk>,
        # no zero division will be encountered
        counts_links /= counts_links.sum(axis=1, dtype=np.float32).reshape(
            (n_pos, 1)
        )
        self.freqs_links = counts_links

        return self
    
    def _predict(self, lemma: str) -> Compound:

        # predict a single lemma and return a DECOW16-format `Compound`
        raw = ""    # will iteratively restore DECOW16-format raw compound
        lemma = f'>{lemma.lower()}<'        # BOS and EOS
        n = self.n
        l = len(lemma) - 1  # -1 because indexing starts at 0
        c = 0   # correction to skip links (see below)

        # same sliding window
        for i in range(1 - n + 1, l - n):  # 1 from both sides because there can not be a link right after BOS
            s = max(0, i + c)   # start of left
            m = i + n + c   # end of left = start of mid
            # break cycle if case left forces the right to be the single EOS
            # which it makes no sense to record because link can not appear there or any further
            if m > l - 1: break # -1 for special symbols
            link_candidates = []
            for r in range(4):  # max length of a link representation is 3 as in -ens-
                e = m + r   # end of mid = start of right
                f = m + r + n   # end of right
                # break cycle if case right context is going to be the single EOS
                if e > l - 1: break   # -1 for special symbols
                left = lemma[s: m]
                mid = lemma[m: e]
                right = lemma[e: f]
                # will return unknown id if unknown
                position_id = self.vocab_positions.encode([left, right, mid])
                candidates = self.freqs_links[position_id]
                link_candidates.append(candidates)
            # not argmax because argmax returns left-most values
            # so if equal probs are in the candidates (e.g 0.5 and 0.5), the left-most is always returned
            link_candidates = np.stack(link_candidates)
            max_freq = link_candidates.max()
            # rows are just observations and are irrelevant;
            # the column index is what we need
            _, best_link_ids = np.where(link_candidates == max_freq)
            best_link_id = random.choice(best_link_ids)

            # top up the raw
            raw += lemma[s + 1] # 1 for BOS

            # if there is no link, then unknown id is returned
            if best_link_id != self.vocab_links.unk_id:
                best_link = self.vocab_links.decode(best_link_id)
                # TODO: formulate better
                # # it turned out the Ngram implementation does not benefit from eliminating allomorphy;
                # # that is easily explainable: the model depend on the concrete ngrams it sees,
                # # and it cannot abstract things; that is why, for example, if you train it that
                # # there is only _+s_ both in -s- and -es- cases,
                # # it will be able to generate _+s_ when it sees XXesXX
                # # but will not be able to deduct -es- from _+s_ because
                # # it can for example refer to -esX- or to -e- when deciding
                # # there is an _+s_ there
                component, realization, link_type = Compound.get_link_info(best_link, eliminate_allomophy=False)
                if link_type == "addition_umlaut":
                    raw = Compound.reverse_umlaut(raw)
                elif link_type == "deletion":
                    to_delete = Compound.get_deletion(component)
                    raw += to_delete
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
                c += len(realization)
                # skip the link

        raw += right[0] # complete raw when the window has sled

        pred = Compound(raw)

        return pred

    def predict(self, compounds: List[str]) -> List[Compound]:

        """
        Make prediction from lemmas to DECOW16-format `Compound`s

        Parameters
        ----------
        compounds : `List[str]`
            lemmas to predict

        Returns
        -------
        `List[Compound]`
            preds in COW format

        Note
        ----
        `NGramsSplitter` does not predict the nominality of stems
        """

        if self.verbose:
            self.progress_bar = tqdm(
                total=len(compounds),
                # leave=True,
                desc="Predicting compounds"
            )
        preds = []
        for compound in compounds:
            pred = self._predict(compound)
            preds.append(pred)
            if self.verbose:
                self.progress_bar.update()
        return preds


def run_baseline(
    gecodb_path: str,
    min_count: Optional[int]=25,
    **params
) -> dict:
    
    """
    Read COW dataset, split it into train and test splits, fit the train split to the n-gram model,
    predict on the test split, evaluate the prediction, and return alongside additional information.

    Parameters
    ----------
    gecodb_path : `str`
        path to the TSV dataset

    min_count : `int`, optional, defaults to 25
        minimal count of compounds to keep; all compounds occurring less will be dropped

    train_split : `float`, optional, defaults to 0.85
        size of the train split

    shuffle : `bool`, optional, defaults to `True`
        whether to shuffle data before splitting

    **kwargs
        parameters to pass to `NGramsSplitter`

    Returns
    -------
    `dict`
        dictionary with the following fields
        * "params": `dict`: passed parameters (for tracking)
        * "train_size": `int`: number of train compounds
        * "density": `float`: density of the compound mask matrix
        * "scores": `dekor.eval.evaluation.EvaluationResult`: metric scores
    """
    
    gecodb = parse_gecodb(gecodb_path, min_count=min_count)

    train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
    train_compounds = train_data["compound"].values
    test_compounds = test_data["compound"].values
    test_lemmas = [
        compound.lemma for compound in test_compounds
    ]

    splitter = NGramsSplitter(**params).fit(train_compounds)
    evaluator = CompoundEvaluator()

    pred_compounds = np.array(splitter.predict(test_lemmas))
    test_data["pred"] = pred_compounds

    scores = evaluator.evaluate(test_compounds, pred_compounds)
    output = {
        'params': params,
        'min_count': min_count,
        'train_size': len(train_data),
        'scores': scores,
        'gold': test_compounds,
        'pred': pred_compounds
    }

    return output


if __name__ == '__main__':

    start = time.time()

    # benchmarking
    in_path = './resources/gecodb_v04.tsv'
    out_dir = './benchmarking/ngrams'

    # param grid
    min_counts = [1000, 100, 20]  # by 10 already out of memory
    ngramss = [2, 3, 4]
    record_none_linkss = [True, False]

    # benchmarking
    outputs = []
    for min_count, ngrams, record_none_links in product(
        min_counts, ngramss, record_none_linkss
    ):
        
        params = {
            'min_count': min_count,
            'n': ngrams,
            'record_none_links': record_none_links,
        }
        
        # print out params
        print('\n', ', '.join(f'{key}: {value}' for key, value in params.items()))
        output = run_baseline(
            in_path,
            verbose=True,
            **params
        )

        outputs.append(output)

    def _sum_scores(scores):
        return sum(scores.values())

    outputs = sorted(outputs, key=lambda entry: _sum_scores(entry['scores']), reverse=True)
    with open(os.path.join(out_dir, 'ngrams.json'), 'w') as out:
        json.dump(outputs, out, indent=4, default=lambda x: "not serializable")

    golds = outputs[0]['gold']
    best_preds = outputs[0]['pred']

    pairs = pd.DataFrame(
        data={
            'gold': [compound.raw for compound in golds],
            'pred': [compound.raw for compound in best_preds]
        }
    )
    pairs.to_csv(
        os.path.join(out_dir, 'ngrams_preds.tsv'),
        sep='\t',
        index=False,
        header=False
    )

    end = time.time()

    print(f"Execution time: {round(end - start, 2)} s") # ~1300s