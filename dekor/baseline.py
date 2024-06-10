import re
from itertools import product
import random
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List

from dekor.gecodb_parser import (
    Compound,
    Link,
    get_span,
    parse_gecodb,
    UMLAUTS_REVERSED
)


class StringVocab:

    def __init__(self):
        self._vocab = {"<unk>": 0}   # unk
        self._vocab_reversed = {0: "unk"}

    def add(self, string):
        if not string in self._vocab:
            id = len(self)
            self._vocab[string] = id
            self._vocab_reversed[id] = string

    def encode(self, ngram):
        return self._vocab.get(ngram, 0)
    
    def decode(self, id):
        return self._vocab_reversed.get(id, "<unk>")
    
    def __len__(self):
        return len(self._vocab)


class NGramsSplitter:

    _empty_link = Link.empty()

    def __init__(self, n=3, accumulative=True, special_symbols=True):
        self.n = n
        self.accumulative = accumulative
        self.max_step = self.n if self.accumulative else 1
        self.special_symbols = special_symbols
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
                self.vocab_ngrams.add(start)
                self.vocab_ngrams.add(end)
                self.vocab_links.add(link.component)
                self.vocab_types.add(link.type)
                # add mask
                masks.append((start, end, link))
        return masks

    def fit(self, compounds: List[Compound]):
        all_masks = []
        for compound in compounds:
            masks = self._forward(compound)
            all_masks += masks
        # at this point, all the vocabs are populated
        freq_links = np.zeros(  # TODO: add-1 smoothing?
            shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links))
        )
        freq_types = np.zeros(  # TODO: add-1 smoothing?
            shape=(len(self.vocab_links), len(self.vocab_types))
        )
        for start, end, link in all_masks:
            freq_links[
                self.vocab_ngrams.encode(start),
                self.vocab_ngrams.encode(end),
                self.vocab_links.encode(link.component)
            ] += 1
            freq_types[
                self.vocab_links.encode(link.component),
                self.vocab_types.encode(link.type)
            ] += 1
        # stochastic probs where applicable
        self.probs_links = freq_links / freq_links.sum(axis=2).reshape(len(self.vocab_ngrams), -1, 1)
        self.probs_types = freq_types / freq_types.sum(axis=1).reshape(-1, 1)
        # fill NaNs
        self.probs_links = np.nan_to_num(self.probs_links, 0)
        self.probs_types = np.nan_to_num(self.probs_types, 0)
        return self

    def _unfuse(self, raw, component, type: str):
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
                raw += '_'
            case _:
                pass
        return raw
    
    def _refine(self, raw):
        # the only case a complex link cannot be restores by removing _-duplicates is addition with umlaut
        raw = re.sub('_+=__+', '_+=', raw)
        raw = re.sub('__', '_', raw)
        return raw

    def _predict(self, compound):
        raw = ""
        lemma = compound.lemma.lower()
        c = 0
        if self.special_symbols:
            lemma = f'>{lemma}<'
            c = 1
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
                if link_candidates.sum():
                    # not argmax because argmax returns left-most values
                    # so if two equal probs are in the candidates, the left-most is always returned
                    best_link_ids = np.where(link_candidates == link_candidates.max())[0]
                    link_ids.append(best_link_ids)
                    link_probs.append(link_candidates.max())

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
            # because of the first 1-, 2-, ..., n-1-grams, first ends will start at the beginning of the word
            if i < (len(lemma) - (self.n + c)): # c for special symbol
                raw += end[0]
            raw = self._unfuse(raw, component, type)
        raw = self._refine(raw)
        return raw

    def predict(self, compounds):
        preds = []
        for compound in compounds:
            pred = self._predict(compound)
            preds.append(pred)
        return preds


def run_baseline(
    gecodb_path,
    min_freq=100000,
    train_split=0.85,
    shuffle=True,
    ngrams=3,
    accumulative=True,
    special_symbols=True
):
    gecodb = parse_gecodb(gecodb_path, min_freq=min_freq)
    train_data, test_data = train_test_split(gecodb, train_size=train_split, shuffle=shuffle)
    train_compounds = train_data["compound"].values
    test_compounds = test_data["compound"].values
    splitter = NGramsSplitter(
        n=ngrams,
        accumulative=accumulative,
        special_symbols=special_symbols
    ).fit(train_compounds)
    preds = splitter.predict(test_compounds)
    return preds


if __name__ == '__main__':
    path = './dekor/COW/gecodb/gecodb_v01.tsv'
    preds = run_baseline(path)
    pass