from src.data.dataset import download_dataset, import_queries, import_collection
import pandas as pd
from tqdm import tqdm
from src.preprocessing.preprocessing import tokenization, removal, stemming
import numpy as np
import logging
from src.embeddings.tfidf import TFIDF
from src.embeddings.elmo import Elmo

tqdm.pandas()
LOGGER = logging.getLogger('pipeline')


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    collection = None
    queries = None
    preprocessed = False

    def __init__(self, collection: str = None, queries: str = None):
        if collection is not None:
            self.collection = pd.read_pickle(collection)
        if queries is not None:
            self.queries = pd.read_pickle(queries)

    def download(self, dataset: str = 'collection'):
        location = download_dataset(dataset)
        self.collection = import_collection(location)
        self.queries = import_queries(location)

    def preprocess(self):
        LOGGER.info('Preprocessing collection')
        self.collection.Passage = self.collection.Passage.progress_apply(lambda text: np.array(
            stemming(
                removal(
                    tokenization(text)
                ))))

        LOGGER.info('Preprocessing queries')
        self.queries.Query = self.queries.Query.progress_apply(lambda text: np.array(
            stemming(
                removal(
                    tokenization(text)
                ))))
        self.preprocessed = True

    def tfidf(self):
        if self.preprocessed is False:
            self.preprocess()

        tfidf = TFIDF()
        self.collection['tfidf_embedding'] = tfidf.fit(self.collection.Passage).transform(self.collection.Passage)

    def elmo(self):
        if self.preprocessed is False:
            self.preprocess()
        elmo = Elmo()
        self.collection['elmo_embedding'] = elmo.fit_transform(self.collection.Passage)
