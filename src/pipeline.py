from src.data.dataset import import_val_test_queries, import_queries, import_collection, import_qrels, \
    import_training_set
import pandas as pd
from tqdm import tqdm
from src.data.preprocessing import preprocess
from src.features.generator import create_bert_embeddings, create_bert_feature, create_glove_feature, \
    create_glove_embeddings, create_w2v_embeddings, create_w2v_feature, create_tfidf_embeddings, create_all, \
    create_BM2_feature, create_tfidf_feature, create_jaccard_feature, create_POS_features, \
    create_interpretation_features, create_sentence_features
import logging
import os
from src.utils.utils import check_path_exists
from src.models.training import Evaluation
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.models.ranknet import RankNet
import torch

tqdm.pandas()
LOGGER = logging.getLogger('pipeline')


class Pipeline(object):
    """ Class to combine the different download, preprocessing, modeling and evaluation steps. """

    collection = None
    queries = None
    queries_val = None
    queries_test = None
    qrels_val = None
    qrels_test = None
    features = pd.DataFrame()
    features_test = pd.DataFrame()
    features_val = pd.DataFrame()

    def __init__(self, collection: str = None, queries: str = None, queries_val: str = None, queries_test: str = None,
                 features: str = None, qrels_val: str = None, qrels_test: str = None, features_test: str = None,
                 features_val: str = None):
        if qrels_val is not None:
            self.qrels_val = pd.read_pickle(qrels_val)
        if qrels_test is not None:
            self.qrels_test = pd.read_pickle(qrels_test)
        if collection is not None:
            self.collection = pd.read_pickle(collection)
        if queries is not None:
            self.queries = pd.read_pickle(queries)
        if queries_val is not None:
            self.queries_val = pd.read_pickle(queries_val)
        if queries_test is not None:
            self.queries_test = pd.read_pickle(queries_test)
        if features is not None:
            self.features = pd.read_pickle(features)
        if features_test is not None:
            self.features_test = pd.read_pickle(features_test)
        if features_val is not None:
            self.features_val = pd.read_pickle(features_val)

    def setup(self, qrel_sampling: int = 20, training_sampling: int = 200, irrelevant_sampling: int = 0,
              datasets: list = None, path: str = 'data/TREC_Passage'):
        if datasets is None:
            datasets = ['collection.tsv', 'queries.train.tsv', 'msmarco-test2019-queries.tsv', '2019qrels-pass.txt',
                        '2020qrels-pass.txt', 'qidpidtriples.train.full.2.tsv', 'msmarco-test2020-queries.tsv']

        if '2019qrels-pass.txt' or '2019qrels-pass.txt' in datasets:
            self.qrels_val, self.qrels_test = import_qrels(path, qrel_sampling)
        if 'msmarco-test2019-queries.tsv' or 'msmarco-test2020-queries.tsv' in datasets:
            self.queries_val, self.queries_test = import_val_test_queries(path, list(self.qrels_val['qID']),
                                                                          list(self.qrels_test['qID']))
        if 'qidpidtriples.train.full.2.tsv' in datasets:
            self.features = import_training_set(path, training_sampling)
        if 'queries.train.tsv' in datasets:
            self.queries = import_queries(path, list(self.features['qID']))
        if 'collection.tsv' in datasets:
            self.collection = import_collection(path, list(self.qrels_val['pID']), list(self.qrels_test['pID']),
                                                list(self.features['pID']), irrelevant_sampling)

        self.queries_val = self.queries_val[self.queries_val['qID'].isin(self.qrels_val['qID'])].reset_index(
            drop=True)
        self.queries_test = self.queries_test[self.queries_test['qID'].isin(self.qrels_test['qID'])].reset_index(
            drop=True)

        return self

    def preprocess(self, expansion=False):
        LOGGER.info('Preprocessing collection')
        self.collection['preprocessed'] = preprocess(self.collection.Passage)

        LOGGER.info('Preprocessing queries')
        self.queries['preprocessed'] = preprocess(self.queries.Query, expansion)

        LOGGER.info('Preprocessing validation queries')
        self.queries_val['preprocessed'] = preprocess(self.queries_val.Query, expansion)

        LOGGER.info('Preprocessing test queries')
        self.queries_test['preprocessed'] = preprocess(self.queries_test.Query, expansion)

        return self

    def create_tfidf_embeddings(self):
        assert self.collection['preprocessed'] is not None, "Preprocess the data first"

        tfidf, self.collection = create_tfidf_embeddings(self.collection, name='collection')
        tfidf, self.queries = create_tfidf_embeddings(self.queries, tfidf=tfidf, name='query')
        tfidf, self.queries_val = create_tfidf_embeddings(self.queries_val, tfidf=tfidf, name='query_val')
        tfidf, self.queries_test = create_tfidf_embeddings(self.queries_test, tfidf=tfidf, name='query_test')

        return self

    def create_w2v_embeddings(self):
        assert self.collection['preprocessed'] is not None, "Preprocess the data first"

        w2v, self.collection = create_w2v_embeddings(self.collection, name='collection')
        w2v, self.queries = create_w2v_embeddings(self.queries, w2v=w2v, name='query')
        w2v, self.queries_val = create_w2v_embeddings(self.queries_val, w2v=w2v, name='query_val')
        w2v, self.queries_test = create_w2v_embeddings(self.queries_test, w2v=w2v, name='query_test')

        return self

    def create_w2v_feature(self, path_collection: str = 'data/embeddings/w2v_collection_embeddings.pkl',
                           path_query: str = 'data/embeddings/w2v_query_embeddings.pkl'):
        self.features = create_w2v_feature(self.features, self.collection, self.queries, path_collection, path_query)

        return self

    def create_bert_embeddings(self):

        bert, self.collection = create_bert_embeddings(self.collection, name='collection')
        bert, self.queries = create_bert_embeddings(self.queries, bert=bert, name='query')
        bert, self.queries_val = create_bert_embeddings(self.queries_val, bert=bert, name='query_val')
        bert, self.queries_test = create_bert_embeddings(self.queries_test, bert=bert, name='query_test')

    def create_glove_embeddings(self):
        assert self.collection['preprocessed'] is not None, "Preprocess the data first"

        glove, self.collection = create_glove_embeddings(self.collection, name='collection')
        glove, self.queries = create_glove_embeddings(self.queries, glove=glove, name='query')
        glove, self.queries_val = create_glove_embeddings(self.queries_val, glove=glove, name='query_val')
        glove, self.queries_test = create_glove_embeddings(self.queries_test, glove=glove, name='query_test')

        return self

    def create_tfidf_feature(self, path_collection: str = 'data/embeddings/tfidf_collection_embeddings.pkl',
                             path_query: str = 'data/embeddings/tfidf_query_embeddings.pkl'):
        self.features = create_tfidf_feature(self.features, self.collection, self.queries, path_collection, path_query)

        return self

    def create_bert_feature(self, path_collection: str = 'data/embeddings/bert_collection_embeddings.pkl',
                            path_query: str = 'data/embeddings/bert_query_embeddings.pkl'):
        self.features = create_bert_feature(self.features, self.collection, self.queries, path_collection, path_query)

        return self

    def create_glove_feature(self, path_collection: str = 'data/embeddings/glove_collection_embeddings.pkl',
                             path_query: str = 'data/embeddings/glove_query_embeddings.pkl'):
        self.features = create_glove_feature(self.features, self.collection, self.queries, path_collection, path_query)

        return self

    def create_jaccard_feature(self):
        self.features = create_jaccard_feature(self.features, self.collection, self.queries)

        return self

    def create_sentence_features(self):
        self.features = create_sentence_features(self.features, self.collection, self.queries)

        return self

    def create_interpretation_features(self):
        self.features = create_interpretation_features(self.features, self.collection, self.queries)

        return self

    def create_POS_features(self):
        self.features = create_POS_features(self.features, self.collection, self.queries)

        return self

    def create_BM25_features(self):
        self.features = create_BM2_feature(self.features, self.collection, self.queries)

        return self

    def create_train_features(self):
        self.features = create_all(self.features, self.collection, self.queries)

        return self

    def create_test_features(self):
        for index, query in self.queries_test.iterrows():
            self.features_test = pd.concat([self.features_test, pd.DataFrame({
                'qID': [query['qID']] * len(self.collection),
                'pID': self.collection['pID']
            })])
        self.features_test = create_all(self.features_test, self.collection, self.queries_test)

    def create_val_features(self):
        for index, query in self.queries_val.iterrows():
            self.features_val = pd.concat([self.features_val, pd.DataFrame({
                'qID': [query['qID']] * len(self.collection),
                'pID': self.collection['pID']
            })])
        self.features_val = create_all(self.features_val, self.collection, self.queries_val)

    def evaluate(self, model: str = 'nb', pca: int = 0,
                 pairwise_model: str = None, pairwise_top_k: int = 50, search_space: list = None,
                 models_path: str = None, store_model_path: str = None):
        evaluation = Evaluation()
        if model == 'nb':
            model_to_test = GaussianNB()
        elif model == 'lr':
            model_to_test = LogisticRegression()
        else:
            model_to_test = MLPClassifier()

        if models_path:
            pairwise_model = torch.load(models_path)
            pairwise_model.eval()
            pairwise_train = False
        else:
            if pairwise_model == 'ranknet':
                pairwise_model = RankNet(len(self.features.columns) - 3)
            else:
                pairwise_model = None
            pairwise_train = True

        if search_space is not None:
            evaluation.hyperparameter_optimization(model_to_test, search_space, self.features,
                                                   self.features_test, self.features_val,
                                                   self.qrels_test, self.qrels_val, 50, pca, pairwise_model, pairwise_top_k, pairwise_train, 50)
        else:
            evaluation(self.features, self.features_test, self.qrels_test, 50, pca, model_to_test, pairwise_model, pairwise_top_k, pairwise_train)

        if store_model_path is not None:
            torch.save(pairwise_model, store_model_path)

    def forward_selection(self, model: str = 'nb', pca: int = 0, search_space: list = None):
        evaluation = Evaluation()
        if model == 'nb':
            model_to_test = GaussianNB()
        elif model == 'lr':
            model_to_test = LogisticRegression()
        else:
            model_to_test = MLPClassifier()

        evaluation.feature_selection(model_to_test, search_space, self.features,
                                     self.features_test, self.features_val,
                                     self.qrels_test, self.qrels_val,
                                     50, pca)

    def save(self, name: str, path: str = 'data/processed'):
        assert name is not None, 'Please provide experiment name'

        check_path_exists(path)
        self.queries.to_pickle(os.path.join(path, name + '_queries.pkl'))
        self.queries_test.to_pickle(os.path.join(path, name + '_queries_test.pkl'))
        self.queries_val.to_pickle(os.path.join(path, name + '_queries_val.pkl'))
        self.collection.to_pickle(os.path.join(path, name + '_collection.pkl'))
        self.features.to_pickle(os.path.join(path, name + '_features.pkl'))
        self.qrels_test.to_pickle(os.path.join(path, name + '_qrels_test.pkl'))
        self.qrels_val.to_pickle(os.path.join(path, name + '_qrels_val.pkl'))
        self.features_test.to_pickle(os.path.join(path, name + '_features_test.pkl'))
        self.features_val.to_pickle(os.path.join(path, name + '_features_val.pkl'))

        return self
