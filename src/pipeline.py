from src.data.dataset import download_dataset, import_queries, import_collection
import pandas as pd


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    collection = None
    queries = None

    def __init__(self, collection: str = None, queries: str = None):
        if collection is not None:
            self.collection = pd.read_pickle(collection)
        if queries is not None:
            self.queries = pd.read_pickle(queries)

    def download_and_import(self, dataset: str = 'collection'):
        location = download_dataset(dataset)
        self.collection = import_collection(location)
        self.queries = import_queries(location)
