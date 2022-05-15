import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
from src.data.preprocessing import pca
from src.utils.utils import save, load, check_path_exists
import os
import json


class Evaluation(object):

    def __init__(self,
                 X_y_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 qrels: pd.DataFrame,
                 k: int = 50,
                 components_pca: int = 0,
                 previous_results: str = 'data/results/results.pkl'):
        self.qrels = qrels
        self.k = k
        self.components_pca = components_pca
        self.previous_results = previous_results

        self.y = X_y_train['y']
        X = X_y_train.drop(columns=['qID', 'pID', 'y'])
        self.test_pair = X_test[['pID', 'qID']]
        X_test = X_test.drop(columns=['pID', 'qID'])

        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(pd.concat([X, X_test])))

        if components_pca > 0:
            data = pca(data, components_pca)

        self.X = data.loc[:len(X) - 1]
        self.X_test = data.loc[len(X):]

        if os.path.exists(previous_results):
            self.results = load(previous_results)
        else:
            check_path_exists(os.path.dirname(previous_results))
            self.results = pd.DataFrame()

    def __call__(self, model=GaussianNB()):
        self.compute_metrics(model)

    def compute_metrics(self, model):
        model.fit(self.X, self.y)
        confidences = pd.DataFrame(model.predict_proba(self.X_test))[1]

        results = pd.DataFrame({
            'confidence': confidences,
            'qID': list(self.test_pair['qID']),
            'pID': list(self.test_pair['pID']),
            'relevant': [0] * len(confidences)
        })
        for i, qrel in self.qrels.iterrows():
            results.loc[((results['pID'] == qrel['pID']) & (results['qID'] == qrel['qID'])), 'relevant'] = qrel[
                'feedback']

        map = self.mean_average_precision_score(results)
        ndcg = self.normalized_discounted_cumulative_gain(results)
        metrics = self.metrics(results)
        k_metrics = self.metrics(results, self.k)

        self.results = pd.concat([self.results,
                                            pd.DataFrame({'model': str(model),
                                                          'hyperparameters': json.dumps(model.get_params()),
                                                          'sampling': len(self.X),
                                                          'pca': self.components_pca,
                                                          'MAP': map,
                                                          'nDCG': ndcg,
                                                          'accuracy': metrics[0],
                                                          'precision': metrics[1],
                                                          'recall': metrics[2],
                                                          'f1': metrics[3],
                                                          f'accuracy@{self.k}': k_metrics[0],
                                                          f'precision@{self.k}': k_metrics[1],
                                                          f'recall@{self.k}': k_metrics[2],
                                                          f'f1@{self.k}': k_metrics[3]
                                                          }, index=[0])]).reset_index(drop=True)

        save(self.results, self.previous_results)

    def calculate_ranks(self, results: pd.DataFrame):
        ranks = results.sort_values('confidence', ascending=False)
        ranks['rank'] = np.arange(1, len(ranks) + 1)
        ranks = ranks[ranks['relevant'] >= 1]
        ranks.index = np.arange(1, len(ranks) + 1)
        return ranks

    def average_precision_score(self, results: pd.DataFrame):
        ranks = self.calculate_ranks(results)

        sum = 0
        for index, data in ranks.iterrows():
            sum += index / data['rank']
        return sum / len(ranks)

    def mean_average_precision_score(self, results: pd.DataFrame):
        qIDs = results['qID'].unique()
        sum = 0
        for qID in qIDs:
            sum += self.average_precision_score(results[results['qID'] == qID])
        return sum / len(qIDs)

    def metrics(self, results: pd.DataFrame, k: int = None):
        if k is not None:
            results = results.sort_values('confidence', ascending=False).groupby('qID').head(k)

        tp = len(results[(results['confidence'] >= 0.5) & (results['relevant'] >= 1)])
        fp = len(results[(results['confidence'] >= 0.5) & (results['relevant'] == 0)])
        tn = len(results[(results['confidence'] < 0.5) & (results['relevant'] == 0)])
        fn = len(results[(results['confidence'] < 0.5) & (results['relevant'] >= 1)])

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = np.nan
        f_score = (2 * precision * recall) / (precision + recall)
        return accuracy, precision, recall, f_score

    def normalized_discounted_cumulative_gain(self, results: pd.DataFrame):
        ranks = self.calculate_ranks(results)

        dcg = 0
        idcg = 0
        for index, data in ranks.sort_values('relevant').reset_index().iterrows():
            dcg += (2 ** data['relevant'] - 1) / np.log2(data['rank'] + 1)
            idcg += (2 ** data['relevant'] - 1) / np.log2((index + 1) + 1)
        return dcg / idcg

    def mean_normalized_discounted_cumulative_gain_score(self, results: pd.DataFrame):
        qIDs = results['qID'].unique()
        sum = 0
        for qID in qIDs:
            sum += self.normalized_discounted_cumulative_gain(results[results['qID'] == qID])
        return sum / len(qIDs)
