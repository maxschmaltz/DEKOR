"""
N-gram model for splitting German compounds based on the COW data.
"""

import re
import json
import time
from itertools import product
import random
import numpy as np
import sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Iterable, Optional, List, Tuple

from dekor.gecodb_parser import (
    Compound,
    Link,
    get_span,
    parse_gecodb,
    UMLAUTS_REVERSED,
    Compound
)
from dekor.eval.evaluate import CompoundEvaluator


class StringVocab:
    
    """
    Simple vocabulary for string occurrences.
    """

    def __init__(self) -> None:
        self._vocab = {"<unk>": 0}
        self._vocab_reversed = {0: "<unk>"}

    def add(self, string: str) -> int:
        
        """
        Adds a string to vocabulary and assigns it a unique id.

        Parameters
        ----------
        string : `str`
            String to add

        Returns
        -------
        `int`
            either the assigned to the newly added string id
            in case the added string was not existent in the vocabulary,
            or the id of the string in case it already existed
        """

        if not string in self._vocab:
            id = len(self)
            self._vocab[string] = id
            self._vocab_reversed[id] = string
        else:
            id = self.encode(string)
        return id

    def encode(self, ngram: str) -> int:

        """
        Get the code of a vocabulary entry

        Parameters
        ----------
        ngram : `str`
            target entry

        Returns
        -------
        `int`
            id of the target entry; returns 0 (code for "<unk>")
            in case the entry is not present in the vocabulary 
        """

        return self._vocab.get(ngram, 0)
    
    def decode(self, id: int) -> str:

        """
        Get the vocabulary entry by its code

        Parameters
        ----------
        id : `id`
            target id

        Returns
        -------
        `str`
            entry of the vocabulary encoded under the target id; returns "<unk>"
            in case the id is unknown to the vocabulary 
        """

        return self._vocab_reversed.get(id, "<unk>")
    
    def __len__(self) -> int:
        return len(self._vocab)


class NGramsSplitter:

    """
    N-grams-based compound splitter that relies on the COW dataset format.
    First, fits train COW entries, then predicts lemma splits in this format.

    Parameters
    ----------
    n : `int`, optional, defaults to `3`
        maximum n-grams length

    accumulative : `bool`, optional, defaults to `True`
        if `True`, will consider n-grams of length `1` to `n`
        inclusively when fitting and predicting compounds,
        e.g. 1-grams, 2-grams, and 3-grams given `n=3`;
        otherwise, will only use n-grams of length `n`

    special_symbols : `bool`, optional, defaults to `True`
        whether to add BOS (">") and EOS ("<") to raw compounds

    verbose : `bool`, optional, defaults to `True`
        whether to show progress bar when fitting and predicting compounds
    """

    _empty_link = Link.empty()

    def __init__(
            self,
            n: Optional[int]=3,
            accumulative: Optional[bool]=True,
            special_symbols: Optional[bool]=True,
            verbose: Optional[bool]=True
        ) -> None:
        self.n = n
        self.accumulative = accumulative
        self.max_step = self.n if self.accumulative else 1
        self.special_symbols = special_symbols
        self.verbose = verbose
        self.vocab_ngrams = StringVocab()
        self.vocab_links = StringVocab()
        self.vocab_types = StringVocab()

    def _forward(self, compound: Compound) -> List[Tuple[int]]:

        # Analyze a single compound; performed as a sliding window
        # over the compound lemma, where for each position it is stored,
        # which left and right context in n-grams there is and
        # whether there is a link it between and, if, yes, which one.
        # Later, uniques context-link pairs will be counted
        # and this counts information will be used in prediction
        lemma = compound.lemma
        c = 0   # indexing correction for special symbols
        if self.special_symbols:
            lemma = f'>{lemma}<'
            c = 1

        # as we know which links to expect, we can track them 
        coming_link_idx = 0
        # masks will be of a form (c_l, c_r, l, t), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * l is the link ("none" if none)
        #   * t is the link type according to the COW dataset
        masks = []
        # Make sliding window; however, we want to start not directly with
        # n-grams, but first come from 1-grams to n-grams at the left of the compound
        # and then slide this n-grams; same with the end: not the last n-gram,
        # but n-gram to 1-gram. To be more clear: having 'Bundestag' and 3-grams, we don't want contexts
        # to be directly (("bun", "des"), ("und", "est"), ..., ("des", "tag")), 
        # but we rather want (("b", "und"), ("bu", "nde"), ("bun", "des"), ..., ("des", "tag"), ("est", "ag"), ("sta", "g")).
        # As we process compounds unidirectionally and move left to right,
        # we want subtract max n-gram length to achieve this effect; thus, with a window of length
        # max n-gram length, we will begin with 1-grams, ..., and end with ..., 1-grams
        for i in range(1 - self.n, (len(lemma) - self.n)):
            # next expected link; we use `_empty_link` to unify the workflow below
            coming_link = compound.links[coming_link_idx] if coming_link_idx < len(compound.links) else self._empty_link
            # Our models can be accumulative or not. If no, only n-grams of length `n` are considered (see example above);
            # otherwise, the same 1-gram to n-gram gain applies not only for the beginning and the end,
            # but for each step, from both ends, for example:
            # (..., ("b", "u"), ("b", "un"), ("b", "und"), ("bu", "u"), ("bu", "nd"), ("bu", "nde"), ("bun", "d"), ...).
            # The workflow in unified with `max_step`: if the model is accumulative,
            # left and right context are windowed additionally to the main window within the inner loop,
            # otherwise `max_step` is `1` so the inner loop runs once with no effect on the main window
            for s_s, s_e in product(range(self.max_step), range(self.max_step)):
                # TODO: if accumulative, masks at the beginning and in the end are duplicated first `n` times
                start = lemma[max(0, i + s_s): i + self.n]    # `s_s` is correction for the first 1-, 2- ... n-1-grams
                end = lemma[i + self.n: i + self.n + self.n - (s_e)] # `s_s` is correction for the last n-1, ..., 2-, 1-grams
                # define if there is a link incoming at this index;
                # `i + self.n`: end of the `start` substring; `c` is correction for special symbols
                if i + self.n == coming_link.span[0] + c:
                    link = coming_link
                    # increment
                    coming_link_idx += 1
                else:
                    link = self._empty_link
                # add the mask
                start_id = self.vocab_ngrams.add(start)
                end_id = self.vocab_ngrams.add(end)
                link_id = self.vocab_links.add(link.component)
                type_id = self.vocab_types.add(link.type)
                masks.append((start_id, end_id, link_id, type_id))

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
                leave=True,
                desc="Fitting compounds"
            )

        all_masks = []
        for compound in compounds:
            # collect masks from a single compound
            masks = self._forward(compound)
            all_masks += masks
            if self.verbose: self.progress_bar.update()
        all_masks = np.array(all_masks)
        n_ngrams, n_links, n_types = len(self.vocab_ngrams), len(self.vocab_links), len(self.vocab_types)

        # at this point, all the vocabs are populated, so we can process masks
        # 1. process link counts
        # a mask has a form (c_l, c_r, l, t), where
        #   * c_l is the left n-gram
        #   * c_r is the right n-gram
        #   * l is the link ("none" if none)
        #   * t is the link type according to the COW dataset
        # we want to get a mapping from <c_l, c_r> to l, so we need first three columns
        masks_links = all_masks[:, :3]
        # count masks
        _, unique_indices, counts = np.unique(masks_links, return_index=True, return_counts=True, axis=0)
        # `np.unique()` sorts masks so the original order is not preserved;
        # to recreate it, the method returns indices of masks in the original dataset
        # in order of their counts; for example, `unique_indices[5]` contains
        # the index of that mask in `masks_links` whose count is under `counts[5]`
        unique_masks_links = masks_links[unique_indices]
        # make `np`-compatible indices: all 0th, all 1st, and then all 2nd dimension indices
        unique_masks_links = unique_masks_links.T
        firsts, seconds, thirds = unique_masks_links   
        # sparse array because the density is too low, usually about 0.5%;
        # we will only consider present links; if upon predicting, no counts
        # at all are present, no link is stated
        counts_links_sparse = sparse.COO(
            [firsts, seconds, thirds],
            counts.astype(np.float32),
            # <c_l, c_r> to l
            shape=(n_ngrams, n_ngrams, n_links)
        )
        # counts to frequencies;
        # the simplest way to do that is to divide the whole thing
        # and then assume that nan values mean no present links were between those contexts
        with np.errstate(divide="ignore", invalid="ignore"):    # ignore divide by zero warning
            counts_links_sparse /= counts_links_sparse.sum(axis=2, dtype=np.float32).reshape(
                (n_ngrams, n_ngrams, 1)
            )
        self.freqs_links_sparse = counts_links_sparse

        # 2. same for types
        # mapping from l to t, so last two columns
        type_links = all_masks[:, 2:4]
        _, unique_indices, counts = np.unique(type_links, return_index=True, return_counts=True, axis=0)
        unique_masks_types = type_links[unique_indices]
        unique_masks_types = unique_masks_types.T
        firsts, seconds = unique_masks_types
        counts_types_sparse = sparse.COO(
            [firsts, seconds],
            counts.astype(np.float32),
            # l to t
            shape=(n_links, n_types)
        )
        with np.errstate(divide="ignore", invalid="ignore"):    # ignore divide by zero warning
            counts_types_sparse /= counts_types_sparse.sum(axis=1, dtype=np.float32).reshape(
                (n_links, 1)
            )
        self.freqs_types_sparse = counts_types_sparse

        return self

    def _unfuse(self, raw: str, next_char: str, component: str, type: str) -> str:
        # iteratively unfuse currently decoded compound part from the incoming link,
        # recreating the COW format
        match type:
            case "addition":
                raw = re.sub(f'{component}$', '', raw, flags=re.I)
                raw += f'_+{component}_'
            case "expansion":
                raw = re.sub(f'{component}$', '', raw, flags=re.I)
                raw += f'_({component})_'
            case "deletion_nom":
                raw += f'{component}_-{component}_'
            case "deletion_non_nom":
                raw += f'{component}_#{component}_'
            case "hyphen":
                raw += '_--_'
            case "umlaut":
                match = re.search('(äu|ä|ö|ü)[^äöü]+$', raw, flags=re.I)
                if match:
                    suffix_after_umlaut = get_span(raw, match.regs[0])
                    umlaut = get_span(raw, match.regs[1])
                    suffix_after_umlaut = re.sub(
                        umlaut,
                        UMLAUTS_REVERSED[umlaut],
                        suffix_after_umlaut,
                        flags=re.I
                    )
                    raw = re.sub(
                        f'{suffix_after_umlaut}$',
                        suffix_after_umlaut,
                        raw,
                        flags=re.I
                    )
                    raw += '_+=_'
            case "concatenation":
                raw += f'_{next_char}'
            case _:
                raw += next_char
        return raw
    
    def _refine(self, raw):
        # fuse double links, e.g. _+xx__-xx_ to _+xx_-xx_ for replacement;
        # the only case a complex link cannot be restores by removing _-duplicates is addition with umlaut,
        # so it comes first
        raw = re.sub('_+=__+', '_+=', raw)
        raw = re.sub('__', '_', raw)
        # remove special symbols
        if self.special_symbols: raw = re.sub('[<>]', '', raw)
        return raw

    def _predict(self, lemma: str) -> Compound:

        # predict a single lemma and return a COW-format `Compound`
        raw = ""    # will iteratively restore COW-format compound
        lemma = lemma.lower()
        if self.special_symbols:
            lemma = f'>{lemma}<'

        # same sliding window
        for i in range(1 - self.n, (len(lemma) - self.n)):

            # for all of the contexts at a certain position (1 if not accumulative),
            # we will store all the freqs and connected links, and will take the most probable one
            link_ids, link_probs = [], []

            # define context
            # This time, we will reverse the slicing so that we retrieve
            # all 1-, 2-, ..., n-grams. We do that because, when we slide the window
            # during fitting, all larger contexts consequently include smaller ones.
            # That's why, if smaller contexts has no connected links, neither will the larger ones,
            # so we can abort the loop earlier
            for s_s, s_e in product(range(self.max_step - 1, -1, -1), range(self.max_step - 1, -1, -1)):

                # TODO: if accumulative, masks at the beginning and in the end are duplicated
                start = lemma[max(0, i + s_s): i + self.n]
                end = lemma[i + self.n: i + self.n + self.n - (s_e)]
                # link frequencies between the contexts
                link_candidates = self.freqs_links_sparse[
                    self.vocab_ngrams.encode(start),
                    self.vocab_ngrams.encode(end)
                ]

                # abort if never occurred
                if not link_candidates.nnz:
                    link_probs = []
                    break

                # not argmax because argmax returns left-most values
                # so if equal probs are in the candidates (e.g 0.5 and 0.5), the left-most is always returned
                link_candidates = link_candidates.todense()
                max_freq = link_candidates.max()
                best_link_ids, = np.where(link_candidates == max_freq)
                link_ids.append(best_link_ids.tolist())
                link_probs.append(max_freq)

            if link_probs:

                # define the most probable link
                link_probs = np.array(link_probs)
                max_link_probs, = np.where(link_probs == link_probs.max())
                best_link_ids = link_ids[random.choice(max_link_probs)] # not argmax once again
                best_link_id = random.choice(best_link_ids)
                # the most probable type
                type_candidates = self.freqs_types_sparse[best_link_id].todense()
                best_type_ids, = np.where(type_candidates == type_candidates.max())
                best_type_id = random.choice(best_type_ids)

                component = self.vocab_links.decode(best_link_id)
                type = self.vocab_types.decode(best_type_id)

            else:   # aborted

                component = ""
                type = "none"

            # restore raw
            raw = self._unfuse(raw, end[0], component, type)

        raw = self._refine(raw)
        pred = Compound(raw)

        return pred

    def predict(self, compounds: List[str]) -> List[Compound]:

        """
        Make prediction from lemmas to COW-format `Compound`s

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
                leave=True,
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
    train_split: Optional[float]=0.85,
    shuffle: Optional[bool]=True,
    **kwargs
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

    train_data, test_data = train_test_split(gecodb, train_size=train_split, shuffle=shuffle)
    train_compounds = train_data["compound"].values
    test_compounds = [
        compound.lemma for compound in test_data["compound"].values
    ]

    splitter = NGramsSplitter(**kwargs).fit(train_compounds)

    preds = splitter.predict(test_compounds)
    test_data["pred"] = preds

    scores = evaluator.evaluate(test_compounds, preds)
    output = {
        'params': params,
        'train_size': len(train_data),
        'density': splitter.freqs_links_sparse.density,
        'scores': scores
    }

    return output


if __name__ == '__main__':

    start = time.time()

    # benchmarking
    in_path = './dekor/COW/gecodb/gecodb_v01.tsv'
    out_path = './benchmarking/ngrams.json'
    evaluator = CompoundEvaluator()

    # param grid
    min_counts = [5000, 1000, 100, 50]  # by 25 already out of memory
    shuffles = [True, False]
    ngramss = [2, 3, 4]
    accumulatives = [True, False]
    special_symbolss = [True, False]

    # testing
    outputs = []
    for min_count, shuffle, ngrams, accumulative, special_symbols in product(
        min_counts, shuffles, ngramss, accumulatives, special_symbolss
    ):
        
        params = {
            'min_count': min_count,
            'shuffle': shuffle,
            'ngrams': ngrams,
            'accumulative': accumulative,
            'special_symbols': special_symbols,
        }
        
        # print out params
        print('\n', ', '.join(f'{key}: {value}' for key, value in params.items()))
        output = run_baseline(
            in_path,
            train_split=0.85,
            verbose=True,
            **params
        )

        outputs.append(output)

    def _sum_scores(scores):
        return sum(scores.values())

    outputs = sorted(outputs, key=lambda entry: _sum_scores(entry['scores']), reverse=True)
    with open(out_path, 'w') as out:
        json.dump(outputs, out, indent=4)

    end = time.time()

    print(f"Execution time: {(end - start).round(2)} s")