# n-grams + classification on link / not + spacy lemmatization of non-links

from nltk import ConditionalFreqDist


class NGramSplitter:

    def __init__(self, n=2, use_special_tokens=True):
        self.n_grams = n
        self.use_special_tokens = use_special_tokens
        