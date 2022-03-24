import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#from trectools import TrecQrel
#import trec_car.read_data

class Preprocessor(object):
   
    def do_removal(texts):
        '''
        Parameters:
        texts: contains 2-dimensional list with texts of all documents

        returns:
        2-dimensional list with texts of all documents where
        - the texts were tokenized
        - punctuation was removed
        (- numbers were removed)
        - and stopwords were removed
        '''

        nltk.download('punkt')
        nltk.download('stopwords')

        all_texts_tokenized = [nltk.word_tokenize(t.lower()) for t in texts]
        all_texts_punct_removed = [[w.translate(str.maketrans('', '', string.punctuation)) for w in t] for t in all_texts_tokenized]

        stopword = set(stopwords.words("english"))
        all_texts_stopwords_removed = [[w for w in t if w not in stopword and w != ''] for t in all_texts_punct_removed] # and not w.isdigit()

        return all_texts_stopwords_removed


    def do_stemm(all_texts):
        '''
        parameters:
        texts: contains 2-dimensional list with texts of all documents

        returns:
        2-dimensional list with texts of all documents where words have been stemmed
        '''

        stemmer = PorterStemmer()

        return [[stemmer.stem(w) for w in t] for t in all_texts]


    def do_lemma(all_texts):
        '''
        parameters:
        texts: contains 2-dimensional list with texts of all documents

        returns:
        2-dimensional list with texts of all documents where words have been lemmatized

        to-do: can be improved with pos tags
        '''

        lemmatizer = WordNetLemmatizer()
        nltk.download('wordnet')

        return [[lemmatizer.lemmatize(w) for w in t] for t in all_texts]


    def do_deriv_norm(all_texts):
        '''
        parameters:
        texts: contains 2-dimensional list with texts of all documents

        might be an idea to try too
        '''

        return "Not implemented"
