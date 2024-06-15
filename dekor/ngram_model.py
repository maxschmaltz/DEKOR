import re
import json
import time
from itertools import product
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Iterable

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

    def __init__(self):
        self._vocab = {"<unk>": 0}   # unk
        self._vocab_reversed = {0: "unk"}

    def add(self, string):
        if not string in self._vocab:
            id = len(self)
            self._vocab[string] = id
            self._vocab_reversed[id] = string
        else:
            id = self.encode(string)
        return id

    def encode(self, ngram):
        return self._vocab.get(ngram, 0)
    
    def decode(self, id):
        return self._vocab_reversed.get(id, "<unk>")
    
    def __len__(self):
        return len(self._vocab)


class NGramsSplitter:

    _empty_link = Link.empty()

    def __init__(self, n=3, accumulative=True, special_symbols=True, verbose=True):
        self.n = n
        self.accumulative = accumulative
        self.max_step = self.n if self.accumulative else 1
        self.special_symbols = special_symbols
        self.verbose = verbose
        self.vocab_ngrams = StringVocab()
        self.vocab_links = StringVocab()
        self.vocab_types = StringVocab()

    def _forward(self, compound):

        lemma = compound.lemma
        c = 0   # correction for special symbols
        if self.special_symbols:
            lemma = f'>{lemma}<'
            c = 1

        coming_link_idx = 0
        masks = []
        for i in range(1 - self.n, (len(lemma) - self.n)):
            # next expected link; _empty link to unify the workflow
            coming_link = compound.links[coming_link_idx] if coming_link_idx < len(compound.links) else self._empty_link
            # define context
            for s_s, s_e in product(range(self.max_step), range(self.max_step)):
                # TODO: if accumulative, masks at the beginning and in the end are duplicated
                start = lemma[max(0, i + s_s): i + self.n]    # correction for the first 1-, 2- ... n-1-grams
                end = lemma[i + self.n: i + self.n + self.n - (s_e)]
                # define if there is a link incoming at this index
                if i + self.n == coming_link.span[0] + c:
                    link = coming_link
                    # increment
                    coming_link_idx += 1
                else:
                    link = self._empty_link
                # add ngrams
                start_id = self.vocab_ngrams.add(start)
                end_id = self.vocab_ngrams.add(end)
                link_id = self.vocab_links.add(link.component)
                type_id = self.vocab_types.add(link.type)
                # add mask
                # masks.append((start, end, link))
                masks.append((start_id, end_id, link_id, type_id))

        return masks

    def fit(self, compounds: Iterable[Compound]):

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
            if self.verbose: self.progress_bar.update()
        all_masks = np.array(all_masks)

        # at this point, all the vocabs are populated
        counts_links = np.zeros(
            shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links))
        )
        counts_types = np.zeros(
            shape=(len(self.vocab_links), len(self.vocab_types))
        )

        # process link counts
        masks_links = all_masks[:, :3]
        _, unique_indices, counts = np.unique(masks_links, return_index=True, return_counts=True, axis=0)
        unique_masks_links = masks_links[unique_indices]
        unique_masks_links = unique_masks_links.T  # all 0th, all 1st, all 2nd dimension indices
        firsts, seconds, thirds = unique_masks_links
        counts_links[firsts, seconds, thirds] = counts
        # take sum of all link counts and thus retrieve left and right contexts
        # between which no links were found
        left, right = np.where(counts_links.sum(axis=2) == 0)
        # set the count of no link to 1 in such cases
        none_link_idx = self.vocab_links.encode("")
        counts_links[left, right, none_link_idx] = 1
        # stochastic probs
        count_sums = counts_links.sum(axis=2)
        self.probs_links = counts_links / np.expand_dims(count_sums, axis=2)

        # same for types
        type_links = all_masks[:, 2:4]
        _, unique_indices, counts = np.unique(type_links, return_index=True, return_counts=True, axis=0)
        unique_masks_types = type_links[unique_indices]
        unique_masks_types = unique_masks_types.T
        firsts, seconds = unique_masks_types
        counts_types[firsts, seconds] = counts
        # types that got no links
        left, = np.where(counts_types.sum(axis=1) == 0)
        # set the count of no link to 1 in such cases
        none_type_idx = self.vocab_types.encode("none")
        counts_types[left, none_type_idx] = 1
        # stochastic probs
        count_sums = counts_types.sum(axis=1)
        self.probs_types = counts_types / np.expand_dims(count_sums, axis=1)

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
            for s_s, s_e in product(range(self.max_step), range(self.max_step)):
                # TODO: if accumulative, masks at the beginning and in the end are duplicated
                start = lemma[max(0, i + s_s): i + self.n]    # correction for the first 1-, 2- ... n-1-grams
                end = lemma[i + self.n: i + self.n + self.n - (s_e)]
                # most probable link
                link_candidates = self.probs_links[
                    self.vocab_ngrams.encode(start),
                    self.vocab_ngrams.encode(end)
                ]
                # not argmax because argmax returns left-most values
                # so if two equal probs are in the candidates, the left-most is always returned
                max_prob = link_candidates.max()
                best_link_ids = np.where(link_candidates == max_prob)[0]
                link_ids.append(best_link_ids.tolist())
                link_probs.append(max_prob)

            if link_probs:
                link_probs = np.array(link_probs)
                max_link_probs = np.where(link_probs == link_probs.max())[0]
                best_link_ids = link_ids[random.choice(max_link_probs)] # not argmax once again
                best_link_id = random.choice(best_link_ids)
                type_candidates = self.probs_types[best_link_id]
                best_type_ids = np.where(type_candidates == type_candidates.max())[0]
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
            if self.verbose: self.progress_bar.update()
        return preds


def run_baseline(
    gecodb_path,
    min_freq=1000,
    train_split=0.85,
    shuffle=True,
    ngrams=3,
    accumulative=True,
    special_symbols=True,
    verbose=True
):
    gecodb = parse_gecodb(gecodb_path, min_freq=min_freq)
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
    # min_freqs = [5000, 1000, 100, 10]
    # shuffles = [True, False]
    # ngramss = [2, 3, 4]
    # accumulatives = [True, False]
    # special_symbolss = [True, False]

    # param grid
    min_freqs = [5000]
    shuffles = [True]
    ngramss = [4]
    accumulatives = [True]
    special_symbolss = [True]

    # testing
    all_scores = []
    for min_freq, shuffle, ngrams, accumulative, special_symbols in product(
        min_freqs, shuffles, ngramss, accumulatives, special_symbolss
    ):
        
        params = {
            'min_freq': min_freq,
            'shuffle': shuffle,
            'ngrams': ngrams,
            'accumulative': accumulative,
            'special_symbols': special_symbols,
        }
        
        # print out params
        print(', '.join(f'{key}: {value}' for key, value in params.items()), '\n')

        try:

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

        except UserWarning:

            all_scores.appendall_scores.append({
                'params': params,
                'train_size': len(train_data),
                'scores': "Out of memory error"
            })
            continue

    def _sum_scores(scores):
        if not isinstance(scores, dict): return -1.0
        return sum(scores.values())

    all_scores = sorted(all_scores, key=lambda entry: _sum_scores(entry['scores']), reverse=True)
    with open(out_path, 'w') as out:
        json.dump(all_scores, out, indent=4)

    end = time.time()

    print(f"Execution time: {(end - start).round(2)} s")