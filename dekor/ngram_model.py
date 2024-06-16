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
        ):
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
        
        """

        if self.verbose:
            self.progress_bar = tqdm(
                total=len(compounds),
                leave=True,
                desc="Fitting compounds"
            )

        all_masks = []
        for compound in compounds:
            masks = self._forward(compound)
            all_masks += masks
            if self.verbose:
                self.progress_bar.update()
        all_masks = np.array(all_masks)

        # at this point, all the vocabs are populated
        # counts_links = np.zeros(
        #     shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links)),
        #     dtype=np.int16
        # )
        # counts_types = np.zeros(
        #     shape=(len(self.vocab_links), len(self.vocab_types)),
        #     dtype=int
        # )

        # process link counts
        masks_links = all_masks[:, :3]
        _, unique_indices, counts = np.unique(masks_links, return_index=True, return_counts=True, axis=0)
        unique_masks_links = masks_links[unique_indices]
        unique_masks_links = unique_masks_links.T  # all 0th, all 1st, all 2nd dimension indices
        firsts, seconds, thirds = unique_masks_links   
        # sparse array because on larger arrays sum crashes
        # counts_links[firsts, seconds, thirds] = counts
        self.counts_links_sparse = sparse.COO(
            [firsts, seconds, thirds],
            counts.astype(np.int64),
            shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links))
        )
        # take sum of all link counts and thus retrieve left and right contexts
        # between which no links were found
        # zero_indicator = (
        #     counts_links_sparse.sum(axis=2, dtype=np.int16) == 0
        # ).astype(np.int8).todense()
        # lefts, rights = np.nonzero(zero_indicator)
        # # set the count of no link to 1 in such cases
        # none_link_idx = self.vocab_links.encode("")
        # # counts_links[left, right, none_link_idx] = 1
        # updated_firsts = np.concatenate((firsts, lefts))
        # updated_seconds = np.concatenate((seconds, rights))
        # updated_thirds = np.concatenate((thirds, np.full(shape=lefts.shape, fill_value=none_link_idx)))
        # updated_counts = np.concatenate((counts, np.ones(shape=lefts.shape)))
        # updated_counts_links_sparse = sparse.COO(
        #     [updated_firsts, updated_seconds, updated_thirds],
        #     updated_counts.astype(np.int16),
        #     shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links))
        # )
        # # stochastic probs
        # count_sums = counts_links_sparse.sum(axis=2, dtype=np.int16)
        # self.probs_links = counts_links_sparse / count_sums

        # same for types
        type_links = all_masks[:, 2:4]
        _, unique_indices, counts = np.unique(type_links, return_index=True, return_counts=True, axis=0)
        unique_masks_types = type_links[unique_indices]
        unique_masks_types = unique_masks_types.T
        firsts, seconds = unique_masks_types
        self.counts_types_sparse = sparse.COO(
            [firsts, seconds],
            counts.astype(np.int64),
            shape=(len(self.vocab_links), len(self.vocab_types))
        )
        # counts_types[firsts, seconds] = counts
        # # types that got no links
        # left, = np.where(counts_types.sum(axis=1) == 0)
        # # set the count of no link to 1 in such cases
        # none_type_idx = self.vocab_types.encode("none")
        # counts_types[left, none_type_idx] = 1
        # # stochastic probs
        # count_sums = counts_types.sum(axis=1)
        # self.probs_types = counts_types / np.expand_dims(count_sums, axis=1)

        return self

    def _unfuse(self, raw, next_char, component, type: str):
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
        # the only case a complex link cannot be restores by removing _-duplicates is addition with umlaut
        raw = re.sub('_+=__+', '_+=', raw)
        raw = re.sub('__', '_', raw)
        if self.special_symbols:
            raw = re.sub('[<>]', '', raw)
        return raw

    def _predict(self, compound):

        raw = ""
        lemma = compound.lemma.lower()
        if self.special_symbols:
            lemma = f'>{lemma}<'

        for i in range(1 - self.n, (len(lemma) - self.n)):

            link_ids, link_probs = [], []

            # define context
            for s_s, s_e in product(range(self.max_step - 1, -1, -1), range(self.max_step - 1, -1, -1)):

                # TODO: if accumulative, masks at the beginning and in the end are duplicated
                start = lemma[max(0, i + s_s): i + self.n]    # correction for the first 1-, 2- ... n-1-grams
                end = lemma[i + self.n: i + self.n + self.n - (s_e)]
                # most probable link
                link_candidates = self.counts_links_sparse[
                    self.vocab_ngrams.encode(start),
                    self.vocab_ngrams.encode(end)
                ]

                if not link_candidates.nnz:
                    link_probs = []
                    break

                # not argmax because argmax returns left-most values
                # so if two equal probs are in the candidates, the left-most is always returned
                link_candidates = link_candidates.todense()
                max_count = link_candidates.max()
                best_link_ids, = np.where(link_candidates == max_count)
                link_ids.append(best_link_ids.tolist())
                link_probs.append(max_count)

            if link_probs:

                link_probs = np.array(link_probs)
                max_link_probs, = np.where(link_probs == link_probs.max())
                best_link_ids = link_ids[random.choice(max_link_probs)] # not argmax once again
                best_link_id = random.choice(best_link_ids)
                type_candidates = self.counts_types_sparse[best_link_id].todense()
                best_type_ids, = np.where(type_candidates == type_candidates.max())
                best_type_id = random.choice(best_type_ids)

                component = self.vocab_links.decode(best_link_id)
                type = self.vocab_types.decode(best_type_id)

            else:

                component = ""
                type = "none"

            # restore raw
            raw = self._unfuse(raw, end[0], component, type)

        raw = self._refine(raw)
        pred = Compound(raw)

        return pred

    def predict(self, compounds):
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
    gecodb_path,
    min_count=100,
    train_split=0.85,
    shuffle=True,
    ngrams=3,
    accumulative=True,
    special_symbols=True,
    verbose=True
):
    gecodb = parse_gecodb(gecodb_path, min_count=min_count)
    train_data, test_data = train_test_split(gecodb, train_size=train_split, shuffle=shuffle)
    train_compounds = train_data["compound"].values
    test_compounds = test_data["compound"].values
    splitter = NGramsSplitter(
        n=ngrams,
        accumulative=accumulative,
        special_symbols=special_symbols,
        verbose=verbose
    ).fit(train_compounds)
    preds = splitter.predict(test_compounds)
    test_data["pred"] = preds
    return train_data, test_data


if __name__ == '__main__':

    start = time.time()

    # benchmarking
    in_path = './dekor/COW/gecodb/gecodb_v01.tsv'
    out_path = './benchmarking/ngrams.json'
    evaluator = CompoundEvaluator()

    # # param grid
    # min_counts = [5000, 1000, 100, 10]
    # shuffles = [True, False]
    # ngramss = [2, 3, 4]
    # accumulatives = [True, False]
    # special_symbolss = [True, False]

    # param grid
    min_counts = [10]
    shuffles = [True, False]
    ngramss = [4]
    accumulatives = [True, False]
    special_symbolss = [True, False]

    # testing
    all_scores = []
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
        print(', '.join(f'{key}: {value}' for key, value in params.items()), '\n')
        train_data, test_data_processed = run_baseline(
            in_path,
            train_split=0.85,
            verbose=True,
            **params
        )

        golds = test_data_processed["compound"].values
        preds = test_data_processed["pred"].values

        scores = evaluator.evaluate(golds, preds)
        all_scores.append({
            'params': params,
            'train_size': len(train_data),
            'scores': scores
        })

    def _sum_scores(scores):
        if not isinstance(scores, dict): return -1.0
        return sum(scores.values())

    all_scores = sorted(all_scores, key=lambda entry: _sum_scores(entry['scores']), reverse=True)
    with open(out_path, 'w') as out:
        json.dump(all_scores, out, indent=4)

    end = time.time()

    print(f"Execution time: {(end - start).round(2)} s")