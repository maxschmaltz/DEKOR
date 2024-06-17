"""
Module for base metric model.
"""

from abc import ABC, abstractmethod
from typing import List

from dekor.gecodb_parser import Compound


class BaseMetric(ABC):

    """
    Base metric class.

    Attributes
    ----------
    name : `str`
        name of the metric
    """

    name: str

    @abstractmethod
    def _calculate(self, gold: Compound, pred: Compound) -> float:
        pass

    def __call__(self, golds: List[Compound], preds: List[Compound]) -> List[float]:

        """
        Calculate metric score.

        Parameters
        ----------
        golds : `List[Compound]`
            gold compounds
        
        preds : `List[Compound]`
            predictions

        Returns
        -------
        `List[float]`
            list of scores for each gold-prediction pair
        """

        scores = [
            self._calculate(gold, pred)
            for gold, pred in zip(golds, preds)
        ]
        return scores