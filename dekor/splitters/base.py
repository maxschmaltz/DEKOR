from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Any, Iterable, List, Self

from dekor.utils.gecodb_parser import Compound


class BaseSplitter(ABC):

    name: str

    @abstractmethod
    def _metadata(self) -> dict:
        pass

    @abstractmethod
    def _forward(self, compound: Compound) -> Any:
        pass

    @abstractmethod
    def fit(self, compounds: Iterable[Compound]) -> Self:
        pass

    @abstractmethod
    def _predict(self, lemma: str) -> Compound:
        pass

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
            preds in DECOW16 compound format
        """

        progress_bar = tqdm(compounds, desc="Predicting") if self.verbose else compounds
        return [
            self._predict(compound)
            for compound in progress_bar
        ]

    def __repr__(self) -> str:
        return str(self._metadata())