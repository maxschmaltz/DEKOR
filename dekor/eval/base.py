from abc import ABC, abstractmethod
from typing import List


class BaseMetric(ABC):

    name: str

    @abstractmethod
    def _calculate(self, gold, pred):
        pass

    def __call__(self, golds, preds):
        scores = [
            self._calculate(gold, pred)
            for gold, pred in zip(golds, preds)
        ]
        return scores