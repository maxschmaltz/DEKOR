import unittest
from sklearn.model_selection import train_test_split

from dekor.gecodb_parser import parse_gecodb
from dekor.models.ngrams import NGramsSplitter


class TestNGramsSplitter(unittest.TestCase):

    """
    In principle, the only thing we can test about the model
    is it's adequacy in the sense that it does not remove
    anything from or add anything from the compounds however it
    splits them.
    """

    def test_lemma_completeness(self):
        gecodb_path = "./resources/gecodb_v04.tsv"
        gecodb = parse_gecodb(gecodb_path, min_count=10000)
        train_data, test_data = train_test_split(gecodb, train_size=0.75, shuffle=False)
        train_compounds = train_data["compound"].values
        test_compounds = test_data["compound"].values
        test_lemmas = [
            compound.lemma for compound in test_compounds
        ]
        splitter = NGramsSplitter(
            n=2,
            record_none_links=False,
            verbose=False
        ).fit(train_compounds)

        pred_compounds = splitter.predict(test_lemmas)
        pred_lemmas = [
            compound.lemma for compound in pred_compounds
        ]
        self.assertListEqual(test_lemmas, pred_lemmas)


if __name__ == '__main__':
    unittest.main()