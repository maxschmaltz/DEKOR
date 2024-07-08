import unittest
import re
from sklearn.model_selection import train_test_split

from dekor.utils.gecodb_parser import parse_gecodb, UMLAUTS_REVERSED
from dekor.splitters import NGramsSplitter, RNNSplitter
from dekor.benchmarking.benchmarking import eval_splitter


class TestNGramsSplitter(unittest.TestCase):

    """
    In principle, the only thing we can test about the model
    is it's adequacy in the sense that it does not remove
    anything from or add anything from the compounds however it
    splits them.
    """

    def test_lemmas_correctness(self):
        gecodb_path = "./resources/gecodb_v04.tsv"
        for eliminate_allomorphy in [True, False]:
            gecodb = parse_gecodb(
                gecodb_path,
                eliminate_allomorphy=eliminate_allomorphy,
                min_count=10000
            )
            train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
            train_compounds = train_data["compound"].values
            test_compounds = test_data["compound"].values
            for n in [2, 3, 4]:
                unfit_splitter = NGramsSplitter(
                    n=n,
                    record_none_links=False,
                    eliminate_allomorphy=eliminate_allomorphy,
                    verbose=False
                )
                _, pred_compounds = eval_splitter(
                    unfit_splitter=unfit_splitter,
                    train_compounds=train_compounds,
                    test_compounds=test_compounds
                )
                test_lemmas = [
                    compound.lemma for compound in test_compounds
                ]
                pred_lemmas = [
                    compound.lemma for compound in pred_compounds
                ]
                self.assertListEqual(test_lemmas, pred_lemmas)


class TestRNNSplitter(unittest.TestCase):

    """
    In principle, the only thing we can test about the model
    is it's adequacy in the sense that it does not remove
    anything from or add anything from the compounds however it
    splits them.
    """

    def test_lemmas_correctness(self):
        gecodb_path = "./resources/gecodb_v04.tsv"
        for eliminate_allomorphy in [True, False]:
            gecodb = parse_gecodb(gecodb_path, eliminate_allomorphy=eliminate_allomorphy, min_count=10000)
            train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
            train_compounds = train_data["compound"].values
            test_compounds = test_data["compound"].values
            test_lemmas = [
                compound.lemma for compound in test_compounds
            ]
            for n in [2, 3, 4]:
                unfit_splitter = RNNSplitter(
                    n=n,
                    eliminate_allomorphy=eliminate_allomorphy,
                    batch_size=4096,
                    verbose=False
                )
                _, pred_compounds = eval_splitter(
                    unfit_splitter=unfit_splitter,
                    train_compounds=train_compounds,
                    test_compounds=test_compounds
                )
                test_lemmas = [
                    compound.lemma for compound in test_compounds
                ]
                pred_lemmas = [
                    compound.lemma for compound in pred_compounds
                ]
                # here, the model might predict umlaut in positions
                # where this is not the case, so we'll need
                # to remove all umlauts to lay off the effect
                test_lemmas_before, pred_lemmas_before = [], []
                for test_lemma, pred_lemma in zip(test_lemmas, pred_lemmas):
                    for after, before in UMLAUTS_REVERSED.items():
                        test_lemma = test_lemma.replace(after, before)
                        pred_lemma = pred_lemma.replace(after, before)
                    test_lemmas_before.append(test_lemma)
                    pred_lemmas_before.append(pred_lemma)
                self.assertListEqual(test_lemmas_before, pred_lemmas_before)


if __name__ == '__main__':
    unittest.main()