import re
import pandas as pd
import numpy as np
from typing import List

from dekor.gecodb_parser import Compound, Link, UMLAUTS_REVERSED, get_span


def parse_gecodb(gecodb_path, min_freq=25):
    gecodb = pd.read_csv(
        gecodb_path,
        sep='\t',
        names=['raw', 'freq'],
        encoding='utf-8'
    )
    if min_freq: gecodb = gecodb[gecodb['freq'] >= min_freq]
    gecodb['compound'] = gecodb['raw'].apply(Compound)
    return gecodb


class StringVocab:

    def __init__(self):
        self._vocab = {"<unk>": 0}   # unk
        self._vocab_reversed = {0: "unk"}

    def add(self, ngram):
        if not ngram in self._vocab:
            id = len(self._vocab)
            self._vocab[ngram] = id
            self._vocab_reversed[id] = ngram

    def encode(self, ngram):
        return self._vocab.get(ngram, 0)
    
    def decode(self, id):
        return self._vocab_reversed.get(id, "<unk>")
    
    def __len__(self):
        return len(self._vocab)


class NGramsSplitter:

    _empty_link = Link.empty()

    def __init__(self, n=3, special_symbols=True):
        self.n = n
        self.special_symbols = special_symbols
        self.vocab_ngrams = StringVocab()
        self.vocab_links = StringVocab()
        self.vocab_types = StringVocab()

    def _forward(self, compound):
        lemma = compound.lemma.lower()
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
            start = lemma[max(0, i): i + self.n]    # correction for the first 1-, 2- ... n-1-grams
            end = lemma[i + self.n: i + self.n + self.n]
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
            # add non-empty links
            masks.append((start, end, link))
        return masks

    def fit(self, compounds: List[Compound]):
        all_masks = []
        for compound in compounds:
            masks = self._forward(compound)
            all_masks += masks
        # at this point, all the vocabs are populated
        freq_links = np.zeros(  # add-1 smoothing?
            shape=(len(self.vocab_ngrams), len(self.vocab_ngrams), len(self.vocab_links))
        )
        freq_types = np.zeros(  # add-1 smoothing?
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

    def _unfuse(self, raw, component, type):
        match type:
            case "addition":
                raw += f'_+{component}_'
            case "addition":
                raw += f'_({component})_'
            case "deletion":
                raw += f'{component}_-{component}_'
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
    
    def refine(self, raw):
        # the only case a complex link cannot be restores by removing _ duplicates is addition with umlaut
        raw = re.sub('_+=__+', '', raw)
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
            # define context
            start = lemma[max(0, i): i + self.n]    # correction for the first 1-, 2- ... n-1-grams 
            end = lemma[i + self.n: i + self.n + self.n]
            # most probable link
            link_candidates = self.probs_links[
                self.vocab_ngrams.encode(start),
                self.vocab_ngrams.encode(end)
            ]
            if link_candidates.sum():
                # not argmax because argmax returns left-most values
                # so if two equal probs are in the candidates, the left-most is always returned
                link_id = np.random.choice(
                    np.flatnonzero(link_candidates == link_candidates.max())
                )
                type_candidates = self.probs_types[link_id]
                type_id = np.random.choice(
                    np.flatnonzero(type_candidates == type_candidates.max())
                )
                component = self.vocab_links.decode(link_id)
                type = self.vocab_types.decode(type_id)
            else:   # no links in observation between contexts
                component = ""
                type = ""
            # restore raw
            # because of the first 1-, 2-, ..., n-1-grams, first ends will start at the beginning of the word
            if i < (len(lemma) - self.n): raw += end[0]
            raw = self._unfuse(raw, component, type)
        return raw

    def predict(self, compounds):
        preds = []
        for compound in compounds:
            pred = self._predict(compound)
            preds.append(pred)
        return preds
            





if __name__ == '__main__':
    path = './dekor/COW/gecodb_v01.tsv'
    gecodb = parse_gecodb(path, min_freq=1000)
    compounds = gecodb.compound.values.tolist()[:5000]
    splitter = NGramsSplitter(n=2).fit(compounds)
    preds = splitter.predict(gecodb.compound.values.tolist()[5000: 5003])
    pass