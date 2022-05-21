import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.decomposition import PCA
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

LOGGER = logging.getLogger('Preprocessor')


def preprocess(data: pd.Series, expansion: bool = False):
    ''' Preprocess Text using tokenization, removing punctuation and stopwords, text expansion, stemming
    
    Args:
        data (pd.Series): Series of strings
        expansion (bool): Decide whether to use word expansion on data or not

    Returns:
        data (pd.Series): Series of np.arrays containing preprocessed text 

    '''
    LOGGER.info('Preprocessing ...')

    if expansion:
        return data.progress_apply(lambda text: np.array(
            stemming(
                query_expansion(
                    removal(
                        tokenization(text)
                    )))))
    else:
        return data.progress_apply(lambda text: np.array(
            stemming(
                removal(
                    tokenization(text)
                ))))


def tokenization(text: str):
    ''' Tokenize using nltk.word_tokenize method and lower string
    
    Args:
        text (str): String of text

    Returns:
        (pd.Series): Series containing lowered tokens 

    '''
    return pd.Series(nltk.word_tokenize(text.lower()))


def removal(tokens: pd.Series):
    ''' Remove punctuation, stopwords and NA values
    
    Args:
        tokens (pd.Series): Series of tokens

    Returns:
        tokens (pd.Series): Series containing tokens with punctuation, stopwords and NA values removed

    '''
    stopwords_list = stopwords.words("english")

    tokens = tokens.apply(lambda token: token.translate(str.maketrans('', '', string.punctuation)))
    tokens = tokens.apply(lambda token: token if token not in stopwords_list and token != '' else None).dropna()

    return tokens


def stemming(tokens: pd.Series):
    ''' Stemm tokens using nltk PorterStemmer method
    
    Args:
        tokens (pd.Series): Series of tokens

    Returns:
        tokens (pd.Series): Series containing stemmed tokens

    '''
    stemmer = PorterStemmer()

    return tokens.apply(lambda token: stemmer.stem(token))


def lemmatization(tokens: pd.Series):
    ''' Lemmatize tokens using nltk WordNetLemmatizer method
    
    Args:
        tokens (pd.Series): Series of tokens

    Returns:
        tokens (pd.Series): Series containing lemmatized tokens

    '''
    lemmatizer = WordNetLemmatizer()

    return tokens.apply(lambda token: lemmatizer.lemmatize(token))


def pca(features: pd.DataFrame, components: int = 5):
    '''
    
    Args:
        features (pd.Series):
        components (int):

    Returns:
        (pd.DataFrame):

    '''
    pca = PCA(components)
    columns = ['pca_comp_%i' % i for i in range(components)]

    return pd.DataFrame(pca.fit_transform(features), columns=columns, index=features.index)


def split_and_scale(X_y_train, X_test, X_val=None, components_pca: int = 0):
    '''
    
    Args:
        X_y_train ():
        X_test ():
        X_val: ():
        components_pca ():

    Returns:
        X ():
        y ():
        X_test ():
        test_pair ():

    '''
    dataframes = []
    y = X_y_train['y']
    X = X_y_train.drop(columns=['qID', 'pID', 'y'])
    dataframes.append(X)
    test_pair = X_test[['pID', 'qID']]
    X_test = X_test.drop(columns=['pID', 'qID'])
    dataframes.append(X_test)

    if X_val is not None:
        val_pair = X_val[['pID', 'qID']]
        X_val = X_val.drop(columns=['pID', 'qID'])
        dataframes.append(X_val)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(pd.concat(dataframes)), columns=X.columns)

    if components_pca > 0:
        data = pca(data, components_pca)

    X = data.loc[:len(X) - 1]
    X_test = data.loc[len(X):len(X) + len(X_test) - 1]

    if X_val is not None:
        X_val = data.loc[len(X) + len(X_test):]
        return X, y, X_test, test_pair, X_val, val_pair

    return X, y, X_test, test_pair


def query_expansion(tokens: pd.Series, sample_size=2):
    ''' Expand series of tokens with synonyms
    
    Args:
        tokens (pd.Series): Series of tokens
        sample_size (int):

    Returns:
        new_tokenlist (pd.Series):

    '''    
    token_list = tokens.tolist()

    new_tokenlist = []
    for token in token_list:
        synonyms = get_synonyms(token, sample_size)

        new_tokenlist.append(token)
        if len(synonyms) > 0:
            new_tokenlist.extend(synonyms)

    return pd.Series(new_tokenlist)


def get_synonyms(phrase, sample_size):
    ''' Create synonyms using wordnets sysnsets method
    
    Args:
        phrase (str):
        sample_size (int):

    Returns:
        synonym_set (list): List containing sysnonyms for given phrase. Only returned if sample_size > set of sysnonyms for phrase
        synonym_set_sampled (list): List containing sampled sysnonyms for given phrase

    '''   
    synonyms = []
    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            if '_' not in l.name() and l.name() != phrase:
                synonyms.append(l.name())

    synonym_set = set(synonyms)

    if sample_size > len(synonym_set):
        return list(synonym_set)
    else:
        synonym_set_sampled = random.sample(synonym_set, sample_size)
        return list(synonym_set_sampled)
