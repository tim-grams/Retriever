import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.decomposition import PCA
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def split_and_scale(X_y_train, X_test, components_pca):
    y = X_y_train['y']
    X = X_y_train.drop(columns=['qID', 'pID', 'y'])
    test_pair = X_test[['pID', 'qID']]
    X_test = X_test.drop(columns=['pID', 'qID'])

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(pd.concat([X, X_test])), columns=X.columns)

    if components_pca > 0:
        data = pca(data, components_pca)

    X = data.loc[:len(X) - 1]
    X_test = data.loc[len(X):]
    return X, y, X_test, test_pair
