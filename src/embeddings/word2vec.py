from gensim.models import Word2Vec
from src.utils.utils import check_path_exists, save, load
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


class word2vec(object):
    ''' A class to create word2vec embeddings.

    Attributes:
        vector_size (int): Dimensionality of the word vectors
        min_count (int): Ignores all words with total frequency lower than this

    Methods:
    fit(text_in_tokens: pd.Series):
        Creates word2vec vocabular and fits model to series of np.arrays containing tokens
    update(new_text_in_tokens): 
        Updates word2vec vocabular and refits model to new series of np.arrays containing tokens
    transform(text_in_tokens: pd.Series, store: str = None):
        Transforms series of preprocessed tokens to word2vec embeddings
    get_wv():
        Returns trained word vectors stored in a KeyedVectors instance
    get_key_vectors():
        Returns dict containing words (str) with corresponding embedding
    vec(word: str):
        Returns word embedding of a specific word
    get_similar(word: str):
        Returns words similar to word
    get_similarity(word1: str, word2: str):
        Returns similarity between two words
        
    '''
    is_fit = False
    is_transform = False

    def __init__(self, vector_size: int = 100, min_count: int = 1):
        ''' Constructs word2vec object with all necessary attributes. 
        
        Args: 
            vector_size (int): Dimensionality of the word vectors
            min_count (int): Ignores all words with total frequency lower than this
        '''
        self.vector_size = vector_size  # Dimensionality of the feature vectors
        self.min_count = min_count  # Ignores all words with lower total absolute frequency

        self.embedding = Word2Vec(vector_size=self.vector_size, window=5, min_count=self.min_count, workers=4)

    def fit(self, text_in_tokens: pd.Series):
        ''' Creates word2vec vocabular and fits model to series of np.arrays containing tokens.
    
        Args:
            text_in_tokens (pd.Series): Series of preprocessed tokens

        Returns:
            none

        '''
        self.embedding.build_vocab(text_in_tokens)
        self.embedding.train(text_in_tokens, total_examples=self.embedding.corpus_count, epochs=self.embedding.epochs)
        self.is_fit = True

        return self

    def update(self, new_text_in_tokens):
        ''' Updates word2vec vocabular and refits model to new series of np.arrays containing tokens.
    
        Args:
            new_text_in_tokens (pd.Series): Series of preprocessed tokens

        Returns:
            none

        '''
        self.embedding.build_vocab(new_text_in_tokens, update=True)
        self.embedding.train(new_text_in_tokens, total_examples=self.embedding.corpus_count, epochs=self.embedding.epochs)

    def transform(self, text_in_tokens: pd.Series, store: str = None):
        ''' Transforms series of preprocessed tokens to word2vec embeddings.
    
        Args:
            new_text_in_tokens (pd.Series): Series of preprocessed tokens
            store (str): Path to store model to

        Returns:
            embeddings (list): list containing np.arrays with word2vec embeddings

        '''
        text_in_tokens = [arr.tolist() for arr in text_in_tokens]
        if self.is_fit is False:
            self.fit(text_in_tokens)
        else:
            self.update(text_in_tokens)

        self.is_transform = True
        
        w = self.get_wv()

        embeddings = []
        missing = []
        for sentence in tqdm(text_in_tokens):
            sen = []
            for word in sentence:
                try:
                    sen.append(w[word])
                except KeyError:
                    #print(word + ' not in vocabular')
                    missing.append(word)
                    sen.append(np.zeros(100))

            embeddings.append(np.array(sen).sum(axis=0))
        #print(str(len(missing)) + ' Unknown words replaced with zero vecs\n')

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(embeddings, store)

        return embeddings

    def transform_tf_idf_weighted(self, text_in_tokens: pd.Series, tf_idf_weights: pd.Series, store: str = None):
        ''' Transforms series of preprocessed tokens to word2vec embeddings with tf/idf weights.
    
        Args:
            new_text_in_tokens (pd.Series): Series of preprocessed tokens
            store (str): Path to store model to

        Returns:
            embeddings (list): list containing np.arrays with word2vec embeddings with tf/idf weights.

        '''
        text_in_tokens = [arr.tolist() for arr in text_in_tokens]
        if self.is_fit is False:
            self.fit(text_in_tokens)
        else:
            self.update(text_in_tokens)

        self.is_transform = True
        
        w = self.get_wv()

        embeddings = []
        missing = []
        for count, sentence in enumerate(tqdm(text_in_tokens)):
            sen = []
            weight_sum = 0
            sen_token_weights = tf_idf_weights[count]

            for word in sentence:
                try:
                    token_weight = sen_token_weights[word]
                    weight_sum += token_weight
                    w2vembedding = w[word]
                    
                    weighted_embedding = token_weight*w2vembedding
                    sen.append(weighted_embedding)
                    
                except KeyError:
                    #print(word + ' not in vocabular')
                    missing.append(word)
                    sen.append(np.zeros(100))

            sen = np.array(sen)
            sen = sen/weight_sum
            embeddings.append(np.array(sen).sum(axis=0))
        #print(str(len(missing)) + ' Unknown words replaced with zero vecs\n')

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(embeddings, store)

        return embeddings

    def get_wv(self):
        ''' Returns trained word vectors stored in a KeyedVectors instance.
    
        Args:
            none

        Returns:
            embedding.wv (KeyedVectors): Trained word vectors stored in a KeyedVectors instance

        '''
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv

    def get_key_vectors(self):
        ''' Returns list of dicts containing words (str) with corresponding embedding.
    
        Args:
            none

        Returns:
            all_vectors (list): list of dicts containing words (str) with corresponding embedding

        '''
        assert self.is_transform is not False, 'You need to use .transform() first'

        all_vectors = []
        for index, vector in enumerate(self.embedding.wv.vectors):
            vector_object = {}
            vector_object[list(self.embedding.wv.key_to_index)[index]] = vector
            all_vectors.append(vector_object)
        return all_vectors

    def vec(self, word: str):
        ''' Returns word embedding of a specific word.
    
        Args:
            word (str): A word as a string

        Returns:
            (np.array): Word embedding of a specific word as np.array
            
        '''
        assert self.is_transform is not False, 'You need to use .transform() first'

        return np.array(self.embedding.wv[word])

    def get_similar(self, word: str):
        ''' Returns words similar to word.
    
        Args:
            word (str): A word as a string

        Returns:
            (list): List conatining similar words and similarity

        '''
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv.most_similar(word)

    def get_similarity(self, word1: str, word2: str):
        ''' Returns similarity between two words.
    
        Args:
            word1 (str): A word as a string
            word2 (str): A word as a string

        Returns:
            (float): Similarity score for word1 and word2 as float

        '''
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv.similarity(word1, word2)
