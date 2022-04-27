import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.decomposition import PCA
import logging
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

LOGGER = logging.getLogger('Preprocessor')


def preprocess(data: pd.Series):
    LOGGER.info('Preprocessing ...')
    return data.progress_apply(lambda text: np.array(
        stemming(
            removal(
                tokenization(text)
            ))))


def tokenization(text: str):
    return pd.Series(nltk.word_tokenize(text.lower()))


def removal(tokens: pd.Series):
    stopwords_list = stopwords.words("english")

    tokens = tokens.apply(lambda token: token.translate(str.maketrans('', '', string.punctuation)))
    tokens = tokens.apply(lambda token: token if token not in stopwords_list and token != '' else None).dropna()

    return tokens


def stemming(tokens: pd.Series):
    stemmer = PorterStemmer()

    return tokens.apply(lambda token: stemmer.stem(token))


def lemmatization(tokens: pd.Series):
    lemmatizer = WordNetLemmatizer()

    return tokens.apply(lambda token: lemmatizer.lemmatize(token))


def pca(features: pd.DataFrame, components: int = 5):
    pca = PCA(components)
    columns = ['pca_comp_%i' % i for i in range(components)]

    return pd.DataFrame(pca.fit_transform(features), columns=columns, index=features.index)
