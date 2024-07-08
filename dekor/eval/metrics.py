"""
Module implementing metrics for evaluation of compound splitters.
"""

from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List

from dekor.utils.gecodb_parser import Compound
from dekor.eval.base import BaseMetric

chencherry = SmoothingFunction(epsilon=0.01)


class CompoundMacroAccuracy(BaseMetric):

    """
    Implements macro accuracy: indicator if gold and prediction are equal.
    Returns 0 or 1.
    """

    name = "macro_accuracy"

    def _calculate(self, gold: Compound, pred: Compound) -> float:
        return float(gold == pred)
    

class CompoundMicroAccuracy(BaseMetric):

    """
    Implements micro accuracy: accuracy of compound components.
    Does not consider alignment but punishes extra/missing components in prediction.
    Returns a float in the interval 0 to 1.
    """

    name = "micro_accuracy" # does not consider alignment but punishes extra preds

    def _calculate(self, gold: Compound, pred: Compound) -> float:
        gold_comps = deepcopy(gold.components)
        pred_comps = deepcopy(pred.components)
        n_matches = 0
        # * punishes missing if `len(gold_comps) > len(pred_comps)`
        # like in zeitpunkt <-- zeitpunkt vs gold zeitpunkt <-- zeit_punkt
        # * punishes extra if `len(gold_comps) < len(pred_comps)`
        # like in belohnungssystem <-- bel_ohnung_+s_system vs gold belohnungssystem <-- belohnung_+s_system
        max_n_components = max(len(gold_comps), len(pred_comps))
        skip_list = []
        for gold_comp in gold_comps:
            for pred_comp in pred_comps:
                if gold_comp in skip_list or pred_comp in skip_list: continue   # to prevent double scoring
                # we could've used the `in` check but it would not consider
                # which fields should and should not be compared
                if gold_comp == pred_comp:
                    n_matches += 1
                    skip_list += [gold_comp, pred_comp] # can't mutate the list during iteration
        return n_matches / max_n_components
    

class CompoundBLEU(BaseMetric):

    """
    Implements BLEU over compound components.
    Returns a float in the interval 0 to 1.
    """

    name = "bleu"
    # Typical linking that we want to recognize includes 3 elements:
    # left part, link, and right part. Therefore, "full" match
    # in a match of 3-grams. That is why we don't use 4-grams and higher,
    # assign weight 2/3 to the "full" 3-grams match, then from the remaining 1/3,
    # we assign the bigger part to 2-grams (which would mean correct link and left/right),
    # and 1-grams get almost no weight because links can be occasional
    weights = (1/18, 5/18, 2/3)   # BLEU-3

    def _get_components(self, comp: Compound) -> List[str]:
        return [
            component.component for component in comp.components
        ]

    def _calculate(self, gold: Compound, pred: Compound) -> float:
        # will only consider components and their alignment
        reference = self._get_components(gold)
        candidate = self._get_components(pred)
        # real example:
        # ['mittel', '_', 'klasse', '_', 'hotel'] vs gold ['mittelklasse', '_', 'hotel']
        # produces BLEU of 
        return sentence_bleu(
            [reference],
            candidate,
            self.weights,
            smoothing_function=chencherry.method2   # add 1 to both numerator and denominator
        )
    

_all_metrics = {
    CompoundMacroAccuracy.name: CompoundMacroAccuracy,
    CompoundMicroAccuracy.name: CompoundMicroAccuracy,
    CompoundBLEU.name: CompoundBLEU
}