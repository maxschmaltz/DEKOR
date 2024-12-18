"""
Module with vocabularies for encoding different information.
"""

from collections import defaultdict


UNK = "<unk>"
UNK_ID = 0


class StringVocab:
    
    """
    Simple vocabulary for string occurrences.
    """

    def __init__(self) -> None:
        self._vocab = {UNK: UNK_ID}
        self._vocab_reversed = {UNK_ID: UNK}
        self.counts = defaultdict(int)

    @property
    def unk(self):
        return UNK

    @property
    def unk_id(self):
        return UNK_ID

    def add(self, string: str) -> int:
        
        """
        Adds a string to vocabulary and assigns it a unique id.

        Parameters
        ----------
        seq : `str`
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
        self.counts[id] += 1
        return id

    def encode(self, string: str) -> int:

        """
        Get the code of a vocabulary entry.

        Parameters
        ----------
        string : `str`
            target entry

        Returns
        -------
        `int`
            id of the target entry; returns unknown id
            in case the entry is not present in the vocabulary 
        """

        return self._vocab.get(string, UNK_ID)
    
    def decode(self, id: int) -> str:

        """
        Get the vocabulary entry by its code.

        Parameters
        ----------
        id : `id`
            target id

        Returns
        -------
        `str`
            entry of the vocabulary encoded under the target id; returns unknown token
            in case the id is unknown to the vocabulary 
        """

        return self._vocab_reversed.get(id, UNK)
    
    def __len__(self) -> int:
        return len(self._vocab)