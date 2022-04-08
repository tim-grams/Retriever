from src.data.dataset import download_dataset, import_queries, import_collection, import_qrels
import pandas as pd
from tqdm import tqdm
from src.preprocessing.preprocessing import tokenization, removal, stemming
import numpy as np
import logging
from src.embeddings.tfidf import TFIDF
from src.features.metrics import cosine_similarity_score
from src.utils.utils import load
import os
from src.utils.utils import check_path_exists

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
        if 'queries.train.tsv' in datasets:
            self.queries = import_queries(path)
        if 'qrels.train.tsv' in datasets:
            self.features['qId'], self.features['pID'] = import_qrels(path)

        self.save()

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

        self.save()

    def create_tfidf_embeddings(self):
        assert self.preprocessed, "Preprocess the data first"

        tfidf = TFIDF()
        self.collection['tfidf_embedding'] = tfidf.fit(
            self.collection['preprocessed']
        ).transform(
            self.collection['preprocessed'],
            "data/embeddings/tfidf_embeddings.pkl")[0]
        self.queries['tfidf_embedding'] = tfidf.transform(self.queries['preprocessed'],
                                                          'data/embeddings/tfidf_embeddings_queries.pkl')

        self.save()

    def create_tfidf_feature(self, path: str = 'data/embeddings'):
        embeddings = load(os.path.join(path, 'tfidf_embeddings.pkl'))
        embeddings_queries = load(os.path.join(path, 'tfidf_embeddings_queries.pkl'))

        self.features['tfidf'] = self.features.progress_apply(lambda qrel:
                                                              cosine_similarity_score(embeddings_queries[qrel.qID],
                                                                                      embeddings[qrel.pID]))

        self.save()

    def save(self, path: str = 'data/processed'):
        check_path_exists(path)
        self.queries.to_pickle(os.path.join(path, '_'.join(self.queries.columns) + '.pkl'))
        self.collection.to_pickle(os.path.join(path, '_'.join(self.collection.columns) + '.pkl'))
        self.features.to_pickle(os.path.join(path, '_'.join(self.features.columns) + '.pkl'))
