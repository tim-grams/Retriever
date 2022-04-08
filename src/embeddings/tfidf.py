from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.utils.utils import check_path_exists, save, load
import os


class TFIDF(object):

    vectorizer = None
    fitted = False

    def __init__(self, path: str = None):
        assert os.path.exists(path), "Vectorizer does not exist"

        self.vectorizer = load(path)

    def fit(self, text_in_tokens: pd.Series, store: str = "models/tfidf.pkl"):
        def dummy(text):
            return text

        self.vectorizer = TfidfVectorizer(tokenizer=dummy, lowercase=False)
        self.vectorizer.fit(text_in_tokens)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(self.vectorizer, store)

        self.fitted = True

        return self

    def transform(self, text_in_tokens: pd.Series, store: str = None):
        assert self.vectorizer is not None, 'You need to fit me first'

        tfidf_matrix = self.vectorizer.transform(text_in_tokens)
        token_names = self.vectorizer.get_feature_names()
        tf_idf_dict = {}
        j = 0

        for name in token_names:
            tf_idf_dict[name] = j
            j += 1

        tf_idf_list = []
        for i in tqdm(range(len(text_in_tokens))):
            tf_idf_token = {}
            for token in text_in_tokens[i]:
                tf_idf_token[token] = tfidf_matrix[i, tf_idf_dict[token]]
            tf_idf_list.append(tf_idf_token)
        tf_idf_vec = np.array(tf_idf_list)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(tfidf_matrix, store)

        return tf_idf_vec


