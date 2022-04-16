from src.data.dataset import download_dataset, import_queries, import_collection, import_qrels
import pandas as pd
from tqdm import tqdm
from src.data.preprocessing import tokenization, removal, stemming
import numpy as np
import logging
from src.embeddings.tfidf import TFIDF
from src.embeddings.glove import Glove
from src.features.features import cosine_similarity_score, euclidean_distance_score, manhattan_distance_score, jaccard, \
    words, relative_difference, characters, difference, subjectivity, polarisation, POS
from src.utils.utils import load
import os
from src.utils.utils import check_path_exists
import time

tqdm.pandas()
LOGGER = logging.getLogger('pipeline')


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    collection = None
    queries = None
    features = pd.DataFrame()

    preprocessed = False

    def __init__(self, collection: str = None, queries: str = None, features: str = None):
        if collection is not None:
            self.collection = pd.read_pickle(collection)
        if queries is not None:
            self.queries = pd.read_pickle(queries)
        if features is not None:
            self.features = pd.read_pickle(features)

    def setup(self, datasets: list = None, path: str = 'data/TREC_Passage'):
        if datasets is None:
            datasets = ['collection.tsv', 'queries.train.tsv', 'qrels.train.tsv']

        download_dataset(datasets)

        if 'collection.tsv' in datasets:
            self.collection = import_collection(path)
        if 'qrels.train.tsv' in datasets:
            self.features['qID'], self.features['pID'] = import_qrels(path, list(self.collection['pID']))
        if 'queries.train.tsv' in datasets:
            self.queries = import_queries(path, list(self.features['qID']))

        return self.save()

    def preprocess(self):
        LOGGER.info('Preprocessing collection')
        self.collection['preprocessed'] = self.collection.Passage.progress_apply(lambda text: np.array(
            stemming(
                removal(
                    tokenization(text)
                ))))

        LOGGER.info('Preprocessing queries')
        self.queries['preprocessed'] = self.queries.Query.progress_apply(lambda text: np.array(
            stemming(
                removal(
                    tokenization(text)
                ))))
        self.preprocessed = True

        return self.save()

    def create_tfidf_embeddings(self):
        assert self.preprocessed, "Preprocess the data first"

        tfidf = TFIDF()
        self.collection['tfidf'] = tfidf.fit(
            self.collection['preprocessed']
        ).transform(
            self.collection['preprocessed'],
            "data/embeddings/tfidf_embeddings.pkl")
        self.queries['tfidf'] = tfidf.transform(self.queries['preprocessed'],
                                                'data/embeddings/tfidf_embeddings_queries.pkl')

        return self.save()

    def create_glove_embeddings(self):
        assert self.preprocessed, "Preprocess the data first"

        glove = Glove()
        self.collection['glove'] = glove.transform(
            self.collection['preprocessed'],
            "data/embeddings/glove_embeddings")
        self.queries['glove'] = glove.transform(self.queries['preprocessed'],
                                                'data/embeddings/glove_embeddings_queries')

        return self.save()

    def create_tfidf_feature(self, path: str = 'data/embeddings'):
        embeddings = load(os.path.join(path, 'tfidf_embeddings.pkl'))
        embeddings_queries = load(os.path.join(path, 'tfidf_embeddings_queries.pkl'))

        self.features['tfidf_cosine'] = self.features.progress_apply(lambda qrel:
                                                                     cosine_similarity_score(embeddings_queries[
                                                                                                 self.queries[
                                                                                                     self.queries[
                                                                                                         'qID'] == qrel.qID].index],
                                                                                             embeddings[self.collection[
                                                                                                 self.collection[
                                                                                                     'pID'] == qrel.pID].index]),
                                                                     axis=1)
        self.features['tfidf_euclidean'] = self.features.progress_apply(lambda qrel:
                                                                        euclidean_distance_score(embeddings_queries[
                                                                                                     self.queries[
                                                                                                         self.queries[
                                                                                                             'qID'] == qrel.qID].index],
                                                                                                 embeddings[
                                                                                                     self.collection[
                                                                                                         self.collection[
                                                                                                             'pID'] == qrel.pID].index]),
                                                                        axis=1)
        self.features['tfidf_manhattan'] = self.features.progress_apply(lambda qrel:
                                                                        manhattan_distance_score(embeddings_queries[
                                                                                                     self.queries[
                                                                                                         self.queries[
                                                                                                             'qID'] == qrel.qID].index],
                                                                                                 embeddings[
                                                                                                     self.collection[
                                                                                                         self.collection[
                                                                                                             'pID'] == qrel.pID].index]),
                                                                        axis=1)

        return self.save()

    def create_jaccard_feature(self):
        self.features['jaccard'] = self.features.progress_apply(
            lambda qrel: jaccard(self.collection[self.collection['pID'] == qrel['pID']]['preprocessed'].iloc[0],
                                 self.queries[self.queries['qID'] == qrel['qID']]['preprocessed'].iloc[0]),
            axis=1)

        return self.save()

    def create_sentence_features(self):
        self.features['words_doc'] = self.features.progress_apply(
            lambda qrel: words(self.collection[self.collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
            axis=1)
        self.features['words_query'] = self.features.progress_apply(
            lambda qrel: words(self.queries[self.queries['qID'] == qrel['qID']]['Query'].iloc[0]),
            axis=1)
        self.features['words_difference'] = self.features.progress_apply(
            lambda qrel: difference(qrel['words_doc'], qrel['words_query']),
            axis=1)
        self.features['words_rel_difference'] = self.features.progress_apply(
            lambda qrel: relative_difference(qrel['words_doc'], qrel['words_query']),
            axis=1)

        self.features['char_doc'] = self.features.progress_apply(
            lambda qrel: characters(self.collection[self.collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
            axis=1)
        self.features['char_query'] = self.features.progress_apply(
            lambda qrel: characters(self.queries[self.queries['qID'] == qrel['qID']]['Query'].iloc[0]),
            axis=1)
        self.features['char_difference'] = self.features.progress_apply(
            lambda qrel: difference(qrel['char_doc'], qrel['char_query']),
            axis=1)
        self.features['char_rel_difference'] = self.features.progress_apply(
            lambda qrel: relative_difference(qrel['char_doc'], qrel['char_query']),
            axis=1)

        return self.save()

    def create_interpretation_features(self):
        self.features['subjectivity_doc'] = self.features.progress_apply(
            lambda qrel: subjectivity(self.collection[self.collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
            axis=1)
        self.features['polarity_doc'] = self.features.progress_apply(
            lambda qrel: polarisation(self.collection[self.collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
            axis=1)

        self.features['subjectivity_query'] = self.features.progress_apply(
            lambda qrel: subjectivity(self.queries[self.queries['qID'] == qrel['qID']]['Query'].iloc[0]),
            axis=1)
        self.features['polarity_query'] = self.features.progress_apply(
            lambda qrel: polarisation(self.queries[self.queries['qID'] == qrel['qID']]['Query'].iloc[0]),
            axis=1)

        return self.save()

    def create_POS_features(self):
        pos = self.features.progress_apply(
            lambda qrel: POS(self.collection[self.collection['pID'] == qrel['pID']]['Passage'].iloc[0]),
            axis=1)
        self.features['doc_nouns'] = [tag[0] for tag in pos]
        self.features['doc_adjectives'] = [tag[1] for tag in pos]
        self.features['doc_verbs'] = [tag[2] for tag in pos]

        pos = self.features.progress_apply(
            lambda qrel: POS(self.queries[self.queries['qID'] == qrel['qID']]['Query'].iloc[0]),
            axis=1)
        self.features['query_nouns'] = [tag[0] for tag in pos]
        self.features['query_adjectives'] = [tag[1] for tag in pos]
        self.features['query_verbs'] = [tag[2] for tag in pos]

        return self.save()

    def save(self, path: str = 'data/processed'):
        check_path_exists(path)
        self.queries.to_pickle(os.path.join(path, 'queries' + str(time.time()) + '.pkl'))
        self.collection.to_pickle(os.path.join(path, 'collection' + str(time.time()) + '.pkl'))
        self.features.to_pickle(os.path.join(path, 'features' + str(time.time()) + '.pkl'))
        return self
