from gensim.models import Word2Vec
import numpy as np


class word2vec(object):
    is_transform = False

    def __init__(self, vector_size, min_count):
        self.vector_size = vector_size  # Dimensionality of the feature vectors
        self.min_count = min_count  # Ignores all words with lower total absolute frequency

        self.embedding = Word2Vec(vector_size=self.vector_size, window=5,
                                  min_count=self.min_count, workers=4)

    def vocabular(self, text_in_tokens):
        self.embedding.build_vocab(text_in_tokens)

    def transform(self, text_in_tokens):
        self.vocabular(text_in_tokens)

        self.embedding.train(text_in_tokens, total_examples=self.embedding.corpus_count, epochs=self.embedding.epochs)
        self.is_transform = True

        return self.embedding.wv.vectors

    # Additional Methods
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
