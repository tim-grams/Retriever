from gensim.models import Word2Vec
from src.utils.utils import check_path_exists, save, load
import numpy as np
import tqdm
import pandas as pd
import os



class word2vec(object):
    is_vocabular = False
    is_transform = False

    def __init__(self, vector_size, min_count):
        self.vector_size = vector_size  # Dimensionality of the feature vectors
        self.min_count = min_count  # Ignores all words with lower total absolute frequency

        self.embedding = Word2Vec(vector_size=self.vector_size, window=5,
                                  min_count=self.min_count, workers=4)

    def vocabular(self, text_in_tokens):
        self.embedding.build_vocab(text_in_tokens)
        # self.is_vocabular = True
        ''' Currently vocab is relearned for each new dataset (collection, query, ...) 
        Solve vocabulary building problem
        Every word that occurs in dataset which is used in the tranform func
        has to be in vocab. -> not all words in the query dataset appear in collection dataset '''

        return self

    def transform(self, text_in_tokens: pd.Series, store: str = None):
        text_in_tokens = [arr.tolist() for arr in text_in_tokens]
        if self.is_vocabular is False: 
            self.vocabular(text_in_tokens)

        self.embedding.train(text_in_tokens, total_examples=self.embedding.corpus_count, epochs=self.embedding.epochs)
        self.is_transform = True

        w = self.get_wv()

        embeddings = []
        for sentence in text_in_tokens:
            sen = []
            for word in sentence:
                sen.append(w[word])
            embeddings.append(np.array(sen).sum(axis=0))
        

        if store is not None:
            check_path_exists(os.path.dirname(store))
            save(embeddings, store)

        return embeddings

    def get_wv(self):
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv

    def get_key_vectors(self):
        assert self.is_transform is not False, 'You need to use .transform() first'

        all_vectors = []
        for index, vector in enumerate(self.embedding.wv.vectors):
            vector_object = {}
            vector_object[list(self.embedding.wv.key_to_index)[index]] = vector
            all_vectors.append(vector_object)
        return all_vectors

    def vec(self, word):
        assert self.is_transform is not False, 'You need to use .transform() first'

        return np.array(self.embedding.wv[word])

    def get_similar(self, word):
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv.most_similar(word)

    def get_similarity(self, word1, word2):
        assert self.is_transform is not False, 'You need to use .transform() first'

        return self.embedding.wv.similarity(word1, word2)