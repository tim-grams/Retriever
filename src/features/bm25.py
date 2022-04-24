import numpy as np
import pandas as pd
from tqdm import tqdm


class BM25(object):

    l_avg = None
    corpus = None
    corpus_length = None
    occurrences = {}

    def fit(self, corpus: pd.Series):
        self.corpus = corpus
        self.l_avg = corpus.apply(lambda passage: passage.size).mean()
        self.corpus_length = self.corpus.size

        for i in tqdm(range(self.corpus.size)):
            for word in self.corpus[i]:
                if word in self.occurrences.keys():
                    self.occurrences[word] += 1
                else:
                    self.occurrences[word] = 1

        return self

    def predict_proba(self, query, document):
        assert self.corpus is not None, 'Fit the model first'

        relevancy = []
        for word in query:
            if word in self.occurrences.keys():
                weight = np.log(0.5 * self.occurrences[word] / self.corpus_length)
                relevancy.append(weight * self.bm25(word, document))

        return sum(relevancy)

    def bm25(self, word, document, k: int = 1, b: float = 0.75):
        term_frequency = np.count_nonzero(document == word)
        l = len(document)
        return (term_frequency * (k + 1)) / (term_frequency + k * l / self.l_avg * b + k * (1 - b))
