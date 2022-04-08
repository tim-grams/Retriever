from src.data.dataset import download_dataset, import_queries, import_collection, import_qrels
import pandas as pd
from src.preprocessing.preprocessing import tokenization, removal, stemming
import numpy as np
import logging
from src.embeddings.tfidf import TFIDF
from src.features.metrics import cosine_similarity_score
from src.utils.utils import load
import os
from src.utils.utils import check_path_exists
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

client = Client()
ProgressBar().register()
LOGGER = logging.getLogger('pipeline')


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    collection = None
    queries = None
    features = dd.from_pandas(pd.DataFrame(), chunksize=25e6)

    preprocessed = False

    def __init__(self, collection: str = None, queries: str = None, features: str = None):
        if collection is not None:
            self.collection = dd.read_csv(collection)
        if queries is not None:
            self.queries = dd.read_csv(queries)
        if features is not None:
            self.features = dd.read_csv(features)

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
        def _preprocess(df):
            return df.apply(lambda text: np.array(
                stemming(
                    removal(
                        tokenization(text)
                    ))))

        LOGGER.info('Preprocessing collection')
        self.collection = self.collection.map_partitions(
            lambda df: df.assign(preprocessed=lambda row: _preprocess(row['Passage'])))

        LOGGER.info('Preprocessing queries')
        self.queries = self.queries.map_partitions(
            lambda df: df.assign(preprocessed=lambda row: _preprocess(row['Query'])))

        self.preprocessed = True
        self.save()

    def create_tfidf_embeddings(self):
        assert self.preprocessed, "Preprocess the data first"

        tfidf = TFIDF()
        collection_data = self.collection['preprocessed'].compute()
        self.collection['tfidf_embedding'] = tfidf.fit(
            collection_data
        ).transform(
            collection_data,
            "data/embeddings/tfidf_embeddings.pkl")[0]
        self.queries['tfidf_embedding'] = tfidf.transform(self.queries['preprocessed'].compute(),
                                                          'data/embeddings/tfidf_embeddings_queries.pkl')

        self.save()

    def create_tfidf_feature(self, path: str = 'data/embeddings'):
        embeddings = load(os.path.join(path, 'tfidf_embeddings.pkl'))
        embeddings_queries = load(os.path.join(path, 'tfidf_embeddings_queries.pkl'))

        self.features = self.features.map_partitions(
            lambda df: df.assign(tfidf=
                                 lambda row: cosine_similarity_score(embeddings[row['pID']],
                                                                     embeddings_queries[row['qID']])))

        self.save()

    def save(self, path: str = 'data/processed'):
        check_path_exists(path)
        self.queries.to_csv(os.path.join(path, '_'.join(self.queries.columns)))
        self.collection.to_csv(os.path.join(path, '_'.join(self.collection.columns)))
        self.features.to_csv(os.path.join(path, '_'.join(self.features.columns)))
