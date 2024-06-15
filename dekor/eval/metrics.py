from copy import deepcopy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from dekor.eval.base import BaseMetric

chencherry = SmoothingFunction(epsilon=0.01)


class CompoundMacroAccuracy(BaseMetric):

    name = "macro_accuracy"

    def _calculate(self, gold, pred):
        return float(gold == pred)
    

class CompoundMicroAccuracy(BaseMetric):

    name = "micro_accuracy" # does not consider alignment but punishes extra preds

    def _calculate(self, gold, pred):
        gold_comps = deepcopy(gold.components)
        pred_comps = deepcopy(pred.components)
        n_matches = 0
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

    name = "bleu"

    def _join_comparable(self, obj):
        comparable_keys = [
            key for key, value in obj.__dataclass_fields__.items()
            if value.compare
        ]
        comparable_values = [
            str(getattr(obj, key))
            for key in comparable_keys
        ]
        return "|".join(comparable_values)

    def _calculate(self, gold, pred):
        reference = [
            self._join_comparable(obj)
            for obj in gold.components
        ]
        candidate = [
            self._join_comparable(obj)
            for obj in pred.components
        ]
        # Typical linking that we want to recognize includes 3 elements:
        # left part, link, and right part. Therefore, "full" match
        # in a match of 3-grams. That is why we don't use 4-grams and higher,
        # assign weight 2/3 to the "full" 3-grams match, then from the remaining 1/3,
        # we assign the bigger part to 2-grams (which would mean correct link and left/right),
        # and 1-grams get almost no weight because links can be occasional
        weights = (1/18, 5/18, 2/3)   # BLEU-3
        return sentence_bleu(
            [reference],
            candidate,
            weights,
            smoothing_function=chencherry.method1   # add epsilon to 0 counts
        )