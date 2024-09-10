"""
Base model for splitting German compounds based on the DECOW16 compound data.
"""

from abc import ABC, abstractmethod
import torch
from typing import Any, Iterable, List, Self

from dekor.utils.gecodb_parser import Compound

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseSplitter(ABC):

    """
    Base class for splitters.
    """

    # caching preprocessed data is impossible because it gets shuffled on every iteration

    name: str

    @abstractmethod
    def _metadata(self) -> dict:
        pass

    @abstractmethod
    def _forward(self, compound: Compound, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def fit(self, compounds: Iterable[Compound], *args, **kwargs) -> Self:
        pass

    @abstractmethod
    def _predict(self, lemma: str, *args, **kwargs) -> Compound:
        pass

    @abstractmethod
    def predict(self, lemmas: List[str], *args, **kwargs) -> List[Compound]:
        pass

    def __repr__(self) -> str:
        return str(self._metadata())