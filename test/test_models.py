import unittest
from sklearn.model_selection import train_test_split

from dekor.utils.gecodb_parser import parse_gecodb
from dekor.splitters.ngrams import NGramsSplitter
from dekor.splitters.nns import FFNSplitter, RNNSplitter
from dekor.benchmarking.benchmarking import eval_splitter


class TestLemmaCorrectness(unittest.TestCase):

    """
    In principle, the only thing we can test about the model
    is it's adequacy in the sense that it does not remove
    anything from or add anything from the compounds however it
    splits them.
    """

    gecodb_path = "./resources/gecodb_v04.tsv"

    def test_ngrams(self):
        # test different parsings
        for eliminate_allomorphy in [True, False]:
            gecodb = parse_gecodb(
                self.gecodb_path,
                eliminate_allomorphy=eliminate_allomorphy,
                min_count=10000
            )
            train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
            train_compounds = train_data["compound"].values
            test_compounds = test_data["compound"].values
            for n in [2, 3, 4]:
                splitter = NGramsSplitter(
                    n=n,
                    record_none_links=False,
                    eliminate_allomorphy=eliminate_allomorphy,
                    verbose=False
                ).fit(train_compounds)
                _, pred_compounds = eval_splitter(
                    splitter=splitter,
                    test_compounds=test_compounds
                )
                test_lemmas = [
                    compound.lemma for compound in test_compounds
                ]
                pred_lemmas = [
                    compound.lemma for compound in pred_compounds
                ]
                self.assertListEqual(test_lemmas, pred_lemmas)

    def test_rnn(self):
        # test different parsings
        for eliminate_allomorphy in [True, False]:
            gecodb = parse_gecodb(
                self.gecodb_path,
                eliminate_allomorphy=eliminate_allomorphy,
                min_count=10000
            )
            train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
            train_compounds = train_data["compound"].values
            test_compounds = test_data["compound"].values
            test_lemmas = [
                compound.lemma for compound in test_compounds
            ]
            for n in [2, 3, 4]:
                splitter = RNNSplitter(
                    n=n,
                    eliminate_allomorphy=eliminate_allomorphy,
                    batch_size=4096,
                    verbose=False
                ).fit(train_compounds)
                _, pred_compounds = eval_splitter(
                    splitter=splitter,
                    test_compounds=test_compounds
                )
                test_lemmas = [
                    compound.lemma for compound in test_compounds
                ]
                pred_lemmas = [
                    compound.lemma for compound in pred_compounds
                ]
                self.assertListEqual(test_lemmas, pred_lemmas)

    def test_ffn(self):
        # test different parsings
        for eliminate_allomorphy in [True, False]:
            gecodb = parse_gecodb(
                self.gecodb_path,
                eliminate_allomorphy=eliminate_allomorphy,
                min_count=10000
            )
            train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=True)
            train_compounds = train_data["compound"].values
            test_compounds = test_data["compound"].values
            test_lemmas = [
                compound.lemma for compound in test_compounds
            ]
            for n in [2, 3, 4]:
                splitter = FFNSplitter(
                    n=n,
                    eliminate_allomorphy=eliminate_allomorphy,
                    batch_size=4096,
                    verbose=False
                ).fit(train_compounds)
                _, pred_compounds = eval_splitter(
                    splitter=splitter,
                    test_compounds=test_compounds
                )
                test_lemmas = [
                    compound.lemma for compound in test_compounds
                ]
                pred_lemmas = [
                    compound.lemma for compound in pred_compounds
                ]
                self.assertListEqual(test_lemmas, pred_lemmas)    


if __name__ == '__main__':
    unittest.main()