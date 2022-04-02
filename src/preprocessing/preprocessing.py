import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


def do_removal(self, texts):
    """
    Parameters:
    texts: contains a list with texts of all documents

    returns:
    2-dimensional list with texts of all documents where
    - the texts were tokenized
    - punctuation was removed
    (- numbers were removed)
    - and stopwords were removed
    """

    nltk.download('punkt')
    nltk.download('stopwords')

    all_texts_tokenized = np.array([nltk.word_tokenize(t.lower()) for t in texts], dtype=object)
    all_texts_punct_removed = np.array([np.array([w.translate(str.maketrans('', '', string.punctuation)) for w in t]) for t in all_texts_tokenized], dtype=object)

    stopword = set(stopwords.words("english"))
    all_texts_stopwords_removed = np.array([np.array([w for w in t if w not in stopword and w != '']) for t in all_texts_punct_removed], dtype=object) # and not w.isdigit()

    return all_texts_stopwords_removed


def do_stemm(self, all_texts):
    '''
    parameters:
    texts: contains a 2-dimensional list with texts of all documents

    returns:
    2-dimensional list with texts of all documents where words have been stemmed
    '''

    stemmer = PorterStemmer()

    return np.array([[stemmer.stem(w) for w in t] for t in all_texts])


def do_lemma(self, all_texts):
    '''
    parameters:
    texts: contains a 2-dimensional list with texts of all documents

    returns:
    2-dimensional list with texts of all documents where words have been lemmatized

    to-do: can be improved with pos tags
    '''

    lemmatizer = WordNetLemmatizer()
    nltk.download('wordnet')

    return np.array([[lemmatizer.lemmatize(w) for w in t] for t in all_texts])


def do_deriv_norm(self, all_texts):
    '''
    parameters:
    texts: contains 2-dimensional list with texts of all documents

    might be an idea to try too
    '''

    return "Not implemented"


def preprocess(self, all_texts, norm_type):
    '''
    parameters:
    deriv_type: "stem", "lemma" (and "norm") that indicate which normalization type to use

    returns: preprocessed documents that are now in two-dimensional arrays
    '''

    removed_texts = self.do_removal(all_texts)

    if (norm_type == "stem"):
        return self.do_stemm(removed_texts)
    elif (norm_type == "lemma"):
        return self.do_lemma(removed_texts)
    #elif (norm_type == "norm"):
    #    return self.do_deriv_norm(removed_texts)
