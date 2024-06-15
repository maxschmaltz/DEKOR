import numpy as np
import pandas as pd
from typing import Dict, List

from dekor.eval.metrics import (
    CompoundMacroAccuracy,
    CompoundMicroAccuracy,
    CompoundBLEU
)

class EvaluationResult(dict):

    def __init__(self, golds, preds, results: Dict[str, List[float]]):
        for metric_name, scores in results.items():
            scores = np.array(scores)
            mean_score = scores.mean(dtype=float)
            self.__setitem__(metric_name, mean_score)
        results['golds'] = golds
        results['preds'] = preds
        self.df = pd.DataFrame(results)
        

class CompoundEvaluator:

    _all_metrics = {
        CompoundMacroAccuracy.name: CompoundMacroAccuracy,
        CompoundMicroAccuracy.name: CompoundMicroAccuracy,
        CompoundBLEU.name: CompoundBLEU
    }

    def __init__(self, metric_names=None):
        self.metrics = []
        if not metric_names:
            metric_names = list(self._all_metrics.keys())
        assert len(metric_names) == len(set(metric_names)), "No duplicates allowed"
        for metric_name in metric_names:
            assert metric_name in self._all_metrics, f"Unknown metric: {metric_name}"
            metric = self._all_metrics[metric_name]()
            self.metrics.append(metric)

    def _run(self, golds, preds):
        results = {}
        for metric in self.metrics:
            metric_values = metric(golds, preds)
            results[metric.name] = metric_values
        res = EvaluationResult(golds, preds, results)
        return res


    def evaluate(self, golds, preds):
        res = self._run(golds, preds)
        return res