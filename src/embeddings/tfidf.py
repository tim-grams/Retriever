from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.utils.utils import check_path_exists, save, load
import os


class TFIDF(object):
    """ A class to create tfidf embeddings.

    Attributes:
        path (str):

    Methods:
    fit(text_in_tokens: pd.Series, store: str = "models/tfidf.pkl"):
        Fits the tfidf model to the data
    transform(text_in_tokens: pd.Series, store: str = None):
        Transforms series of preprocessed tokens to tfidf embeddings

    """

    vectorizer = None
    fitted = False

    def __init__(self, path: str = None):
        """ Constructs tfidf object.

        Args:
            path (str): Path of model

        """
        if path is not None:
            self.vectorizer = load(path)

    def fit(self, text_in_tokens: pd.Series, store: str = "models/tfidf.pkl"):
        """ Fits the tfidf model to the data.

        Args:
            text_in_tokens (pd.Series): Series of preprocessed tokens
            store (str): Path to store model to

        Returns:
            none

        """
        def dummy(text):
            return text

        self.vectorizer = TfidfVectorizer(tokenizer=lambda text: text, lowercase=False)
        self.vectorizer.fit(text_in_tokens)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(self.vectorizer, store)

        self.fitted = True

        return self

    def transform(self, text_in_tokens: pd.Series, store: str = None):
        """ Transform series of preprocessed tokens to tfidf embeddings.

        Args:
            text_in_tokens (pd.Series): Series of preprocessed tokens
            store (str): Path to tfidf embeddings to

        Returns:
            tf_idf_vec (np.array): Array containing tfidf embeddings

        """
        assert self.vectorizer is not None, 'You need to fit me first'

        tfidf_matrix = self.vectorizer.transform(text_in_tokens)
        token_names = self.vectorizer.get_feature_names_out()
        tf_idf_dict = {}
        j = 0

        for name in token_names:
            tf_idf_dict[name] = j
            j += 1

        tf_idf_list = []
        for i in tqdm(range(len(text_in_tokens))):
            tf_idf_token = {}
            for token in text_in_tokens.iloc[i]:
                if token in tf_idf_dict.keys():
                    tf_idf_token[token] = tfidf_matrix[i, tf_idf_dict[token]]
                else:
                    tf_idf_token[token] = .0
            tf_idf_list.append(tf_idf_token)
        tf_idf_vec = np.array(tf_idf_list)

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(tfidf_matrix, store)

        return tf_idf_vec


