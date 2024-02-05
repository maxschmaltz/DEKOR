import pandas
from collections import defaultdict, UserString
import itertools
import re
from typing import List, Dict

from patterns import LINK_TYPES


class Compound:

    def __init__(self, raw_compound: str) -> None:
        self._link_freqs = defaultdict(int)
        self._raw = raw_compound
        self._splitted = self._split_raw_compound(raw_compound)
        self._pretty = self._unscramble_raw_compound(self.splitted)

    @property
    def raw(self) -> str:
        return self._raw
    
    @property
    def splitted(self) -> List[str]:
        return self._splitted
    
    @property
    def pretty(self) -> str:
        return self._pretty
    
    @property
    def link_freqs(self) -> Dict[str, str]:
        return dict(self._link_freqs)

    def _split_raw_compound(
        self,
        raw_compound: str
    ) -> List[str]:
        _raw_compound = raw_compound
        boundaries = []
        for pattern, link_type in LINK_TYPES.items():
            for match in re.finditer(pattern, _raw_compound, flags=re.I):  # ignore case
                start, end = match.span()   # link span
                self._link_freqs[link_type] += 1
                boundaries += [start, end]
                _raw_compound = _raw_compound[:start] + _raw_compound[end:]    # remove link to prevent double match
        splitted_compound = [
            raw_compound[start: end]
            for start, end in itertools.pairwise([0] + boundaries + [len(raw_compound)])
        ]
        return splitted_compound
    
    def _unscramble_raw_compound(self, splitted_compound: List[str]):
        pass



def read_data(gecodb_path, min_freq=None):
    gecodb = pandas.read_csv(
        gecodb_path,
        sep='\t',
        names=['comp', 'freq'],
        encoding='utf-8'
    )
    if min_freq is not None:
        gecodb[gecodb['freq'] >= min_freq]
    return gecodb


    





def type_freqs(gecodb):
    freqs = defaultdict(int)
    gecodb['comp'].apply(_type_freqs, args=(freqs,))
    return dict(freqs)


if __name__ == '__main__':
    for c in ["Gast_+=e_Buch", "Datum_-um_+en_Satz", "Reptil_(i)_+en_Art"]:
        comp = Compound(c)
        pass