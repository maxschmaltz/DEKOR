"""
Module implementing compound splitter evaluator.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from dekor.gecodb_parser import Compound
from dekor.eval.metrics import _all_metrics

class EvaluationResult(dict):

    """
    Evaluation result of compound splitter prediction against gold compounds.

    Attributes
    ----------
    <metric_name> : `float`
        average metric score (retrievable for **each** metric by it's name)

    df : `pandas.DataFrame`
        dataframe with gold, prediction, and each metric score for each entry of the test data,
        retrievable under columns "golds", "preds", <metric_name> respectively
    """

    def __init__(
        self,
        golds: List[Compound],
        preds: List[Compound],
        results: Dict[str, List[float]]
    ) -> None:
        for metric_name, scores in results.items():
            scores = np.array(scores)
            mean_score = scores.mean(dtype=float)
            self.__setitem__(metric_name, mean_score)
        results['golds'] = golds
        results['preds'] = preds
        self.df = pd.DataFrame(results)
        

class CompoundEvaluator:

    """
    Class for computing metric scores for compounds.

    Parameters
    ----------
    metric_names : List[str], optional
        metric names to run; if `None`, all metrics will be used;
        look up names with `CompoundEvaluator.list_metric_names()`
    """

    def __init__(self, metric_names: Optional[List[str]]=None) -> None:
        self.metrics = []
        if not metric_names:
            metric_names = self.list_metric_names()
        assert len(metric_names) == len(set(metric_names)), "No duplicates allowed"
        for metric_name in metric_names:
            assert metric_name in _all_metrics, f"Unknown metric: {metric_name}"
            metric = _all_metrics[metric_name]()
            self.metrics.append(metric)

    @staticmethod
    def list_metric_names() -> List[str]:

        """
        List all available metric names.

        Returns
        -------
        `List[str]`
            metric names
        """

        return list(_all_metrics.keys())

    def _run(self, golds: List[Compound], preds: List[Compound]) -> EvaluationResult:
        results = {}
        for metric in self.metrics:
            metric_values = metric(golds, preds)
            results[metric.name] = metric_values
        res = EvaluationResult(golds, preds, results)
        return res


    def evaluate(self, golds: List[Compound], preds: List[Compound]) -> EvaluationResult:

        """
        Evaluate splitter predictions against gold compounds.

        Parameters
        ----------
        golds : `List[Compound]`
            gold compounds
        
        preds : `List[Compound]`
            predictions

        Returns
        -------
        `EvaluationResult`
            result of evaluation with both element-wise and average metrics
        """

        res = self._run(golds, preds)
        return res