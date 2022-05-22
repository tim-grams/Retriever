import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from src.data.preprocessing import split_and_scale
from src.utils.utils import save, load, check_path_exists
import os
import json
from skopt.utils import use_named_args
from skopt import gp_minimize
from src.models.pairwise import pairwise_optimize


class Evaluation(object):
    ''' A class to create perform model evaluations.

    Attributes:
        previous_results (str): Path to previously stored results

    Methods:
    __call__(X_y_train: pd.DataFrame, X_test: pd.DataFrame, qrels: pd.DataFrame, k: int = 50,
                 components_pca: int = 0, model=GaussianNB(), pairwise_model=None, pairwise_top_k: int = 50,
                 pairwise_train: bool = True, name: str = None, save_result: bool = True):
        INSERT_DESCRIPTION
    hyperparameter_optimization(model, search_space, X_y_train: pd.DataFrame, X_test: pd.DataFrame,
                                    X_val: pd.DataFrame, qrels: pd.DataFrame, qrels_val: pd.DataFrame,
                                    k: int = 50, components_pca: int = 0, pairwise_model=None,
                                    pairwise_top_k: int = 50, pairwise_train: bool = True,
                                    trials: int = 50, name: str = None, save_result: bool = True):
        Performs hyperparameter optimization.
    feature_selection(model, search_space, X_y_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame,
                            qrels: pd.DataFrame, qrels_val: pd.DataFrame, k: int = 50, components_pca: int = 0,
                            save_results: bool = True, name: str = None):
        Performs feature selection.
    compute_metrics(model, X: pd.DataFrame, y, X_test, test_pair, qrels: pd.DataFrame, k: int = 50,
                        components_pca: int = 0, pairwise_model=None, pairwise_top_k: int = 50,
                        pairwise_train: bool = True, name: str = None, save_result: bool = False):
        INSERT_DESCRIPTION
    calculate_ranks(results: pd.DataFrame):
        INSERT_DESCRIPTION
    average_precision_score(results: pd.DataFrame):
        INSERT_DESCRIPTION
    mean_average_precision_score(results: pd.DataFrame):
        INSERT_DESCRIPTION
    metrics(results: pd.DataFrame, k: int = None):
        INSERT_DESCRIPTION
    normalized_discounted_cumulative_gain(results: pd.DataFrame):
        INSERT_DESCRIPTION
    mean_normalized_discounted_cumulative_gain_score(results: pd.DataFrame):
        INSERT_DESCRIPTION
    mean_reciprocal_rank(results: pd.DataFrame):
    
    '''    

    def __init__(self, previous_results: str = 'data/results/results.pkl'):
        ''' Constructs Evaluation object. 
        
        Args: 
            previous_results (str): Path to previously stored resultsl

        '''
        self.previous_results = previous_results

        if os.path.exists(previous_results):
            self.results = load(previous_results)
        else:
            check_path_exists(os.path.dirname(previous_results))
            self.results = pd.DataFrame()

    def __call__(self,
                 X_y_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 qrels: pd.DataFrame,
                 k: int = 50,
                 components_pca: int = 0,
                 model=GaussianNB(),
                 pairwise_model=None,
                 pairwise_top_k: int = 50,
                 pairwise_train: bool = True,
                 name: str = None,
                 save_result: bool = True):
        ''' INSERT_DESCRIPTION.
    
        Args:
            X_y_train (pd.DataFrame):
            X_test (pd.DataFrame):
            qrels (pd.DataFrame):
            k (int): 
            components_pca (int):
            model ():
            pairwise_model (str):
            pairwise_top_k (int):
            pairwise_train (Boolean):
            name (str):
            save_result (Boolean):

        Returns:
            best_result (): 

        '''  
        X, y, X_test, test_pair = split_and_scale(X_y_train, X_test, components_pca=components_pca)
        mrr = self.compute_metrics(model,
                                   X,
                                   y,
                                   X_test,
                                   test_pair,
                                   qrels,
                                   k,
                                   components_pca,
                                   pairwise_model,
                                   pairwise_top_k,
                                   pairwise_train,
                                   name=name,
                                   save_result=save_result)
        print(f'MRR: {mrr}')

    def hyperparameter_optimization(self, model, search_space,
                                    X_y_train: pd.DataFrame,
                                    X_test: pd.DataFrame,
                                    X_val: pd.DataFrame,
                                    qrels: pd.DataFrame,
                                    qrels_val: pd.DataFrame,
                                    k: int = 50,
                                    components_pca: int = 0,
                                    pairwise_model=None,
                                    pairwise_top_k: int = 50,
                                    pairwise_train: bool = True,
                                    trials: int = 50,
                                    name: str = None,
                                    save_result: bool = True
                                    ):
        ''' Performs hyperparameter optimization.
    
        Args:
            model ():
            search_space ():
            X_y_train (pd.DataFrame):
            X_test (pd.DataFrame):
            X_val (pd.DataFrame):
            qrels (pd.DataFrame):
            qrels_val (pd.DataFrame):
            k (int): 
            components_pca (int):
            pairwise_model (str):
            pairwise_top_k (int):
            pairwise_train (Boolean):
            name (str):
            save_result (Boolean):

        Returns:
            best_result (): 

        '''
        @use_named_args(search_space)
        def evaluate(**params):
            model.set_params(**params)
            return self.compute_metrics(model, X, y, X_val, val_pair, qrels_val, k, components_pca, pairwise_model, pairwise_top_k, pairwise_train, name=name)

        X, y, X_test, test_pair, X_val, val_pair = split_and_scale(X_y_train, X_test, X_val, components_pca)
        best_result = gp_minimize(evaluate, search_space, n_calls=trials)
        print(f'Best MRR: {best_result.fun}')
        print(f'Best Hyperparameters: {best_result.x}')

        best_params_dict = {}
        for space, value in zip(search_space, best_result.x):
            best_params_dict[space.name] = value
        print(
            f'MRR on test set: {self.compute_metrics(model.set_params(**best_params_dict), X, y, X_test, test_pair, qrels, k, components_pca, pairwise_model, pairwise_top_k, pairwise_train, name=name, save_result=save_result)}')

        return best_result.fun

    def feature_selection(self, model, search_space,
                          X_y_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          X_val: pd.DataFrame,
                          qrels: pd.DataFrame,
                          qrels_val: pd.DataFrame,
                          k: int = 50,
                          components_pca: int = 0,
                          save_results: bool = True,
                          name: str = None
                          ):
        ''' Performs feature selection.
    
        Args:
            model ():
            search_space ():
            X_y_train (pd.DataFrame):
            X_test (pd.DataFrame):
            X_val (pd.DataFrame):
            qrels (pd.DataFrame):
            qrels_val (pd.DataFrame):
            k (int): 
            components_pca (int):
            name (str):
            save_result (Boolean):

        Returns:
            added_columns (list): 

        '''
        features = list(X_y_train.drop(columns=['qID', 'pID', 'y']).columns)
        added_columns = []

        current_best = (None, 0)
        current_performance = -1
        while len(added_columns) < len(features):
            for feature in features:
                if feature in added_columns:
                    continue
                print(f'Testing features: {added_columns + [feature]}')
                performance = self.hyperparameter_optimization(model,
                                                               search_space,
                                                               X_y_train[
                                                                   ['qID', 'pID', 'y'] + added_columns + [feature]],
                                                               X_test[['qID', 'pID'] + added_columns + [feature]],
                                                               X_val[['qID', 'pID'] + added_columns + [feature]],
                                                               qrels,
                                                               qrels_val,
                                                               k,
                                                               components_pca,
                                                               name=name,
                                                               save_result=save_results)
                if performance > current_performance and performance > current_best[1]:
                    current_best = (feature, performance)
            if current_best[0] is not None:
                current_performance = current_best[1]
                added_columns.append(current_best[0])
            else:
                break
            current_best = (None, 0)
            print(f'Current features: {added_columns}')
            print(f'Current Performance: {current_performance}')

        print(f'Best feature combination: {added_columns}')
        print(f'MRR: {current_performance}')

        return added_columns

    def compute_metrics(self, model,
                        X: pd.DataFrame,
                        y,
                        X_test,
                        test_pair,
                        qrels: pd.DataFrame,
                        k: int = 50,
                        components_pca: int = 0,
                        pairwise_model=None,
                        pairwise_top_k: int = 50,
                        pairwise_train: bool = True,
                        name: str = None,
                        save_result: bool = False):
        ''' Computes metrics.
    
        Args:
            model ():
            x (pd.DataFrame):
            y ():
            test_pair ():
            qrels (pd.DataFrame):
            k (int): 
            components_pca (int):
            pairwise_model (str):
            pairwise_top_k (int):
            pairwise_train (Boolean):
            name (str):
            save_result (Boolean):

        Returns:
            mrr (): 

        '''
        model.fit(X, y)
        confidences = pd.DataFrame(model.predict_proba(X_test))[1]

        results = pd.DataFrame({
            'confidence': confidences,
            'qID': list(test_pair['qID']),
            'pID': list(test_pair['pID']),
            'relevant': [0] * len(confidences)
        })
        for i, qrel in qrels.iterrows():
            results.loc[((results['pID'] == qrel['pID']) & (results['qID'] == qrel['qID'])), 'relevant'] = qrel[
                'feedback']

        if pairwise_model is not None:
            results = pairwise_optimize(pairwise_model, results, X, y, X_test, pairwise_top_k, pairwise_train)

        mrr = self.mean_reciprocal_rank(results)
        map = self.mean_average_precision_score(results)
        ndcg = self.mean_normalized_discounted_cumulative_gain_score(results)
        metrics = self.metrics(results)
        k_metrics = self.metrics(results, k)

        if save_result:
            self.results = pd.concat([self.results,
                                      pd.DataFrame({'name': name,
                                                    'model': str(model),
                                                    'hyperparameters': json.dumps(model.get_params()),
                                                    'pairwise_model': pairwise_model,
                                                    'pairwise_k': pairwise_top_k if pairwise_model is not None else None,
                                                    'features': json.dumps(list(X.columns)),
                                                    'sampling_training': len(X),
                                                    'sampling_test': len(X_test),
                                                    'pca': components_pca,
                                                    'MRR': mrr,
                                                    'MAP': map,
                                                    'nDCG': ndcg,
                                                    'accuracy': metrics[0],
                                                    'precision': metrics[1],
                                                    'recall': metrics[2],
                                                    'f1': metrics[3],
                                                    f'accuracy@{k}': k_metrics[0],
                                                    f'precision@{k}': k_metrics[1],
                                                    f'recall@{k}': k_metrics[2],
                                                    f'f1@{k}': k_metrics[3]
                                                    }, index=[0])]).reset_index(drop=True)
            save(self.results, self.previous_results)

        return mrr

    def calculate_ranks(self, results: pd.DataFrame):
        ''' Calculates ranks.
    
        Args:
            results (pd.DataFrame):

        Returns:
            ranks (pd.DataFrame): 

        '''
        ranks = results.sort_values('confidence', ascending=False)
        print(len(results))
        ranks['rank'] = np.arange(1, len(ranks) + 1)
        ranks = ranks[ranks['relevant'] >= 1]
        ranks.index = np.arange(1, len(ranks) + 1)
        return ranks

    def average_precision_score(self, results: pd.DataFrame):
        ''' Calculates average precision score.
    
        Args:
            results (pd.DataFrame):

        Returns:
            sum / len(ranks) (float): 

        '''
        ranks = self.calculate_ranks(results)
        sum = 0
        for index, data in ranks.iterrows():
            sum += index / data['rank']
        return sum / len(ranks)

    def mean_average_precision_score(self, results: pd.DataFrame):
        ''' Calculates mean average precision score.
    
        Args:
            results (pd.DataFrame):

        Returns:
            sum / len(qIDs) (float): 

        '''        
        qIDs = results['qID'].unique()
        sum = 0
        for qID in qIDs:
            sum += self.average_precision_score(results[results['qID'] == qID])
        return sum / len(qIDs)

    def metrics(self, results: pd.DataFrame, k: int = None):
        ''' Calculates metrics (accuracy, precision, recall, f1).
    
        Args:
            results (pd.DataFrame):
            k (int): 

        Returns:
            accuracy (float): Returns accuracy score of model
            precision (float): Returns precision score of model
            recall (float): Returns recall score of model
            f_score (float): Returns f_score score of model

        ''' 
        if k is not None:
            results = results.sort_values('confidence', ascending=False).groupby('qID').head(k)

        tp = len(results[(results['confidence'] >= 0.5) & (results['relevant'] >= 1)])
        fp = len(results[(results['confidence'] >= 0.5) & (results['relevant'] == 0)])
        tn = len(results[(results['confidence'] < 0.5) & (results['relevant'] == 0)])
        fn = len(results[(results['confidence'] < 0.5) & (results['relevant'] >= 1)])

        accuracy = (tp + tn) / (tp + fp + tn + fn)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = np.nan
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = np.nan
        try:
            f_score = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f_score = np.nan
        return accuracy, precision, recall, f_score

    def normalized_discounted_cumulative_gain(self, results: pd.DataFrame):
        ''' Calculates normalized discounted cumulative gain.
    
        Args:
            results (pd.DataFrame):

        Returns:
            dcg / idcg (float):

        ''' 
        ranks = self.calculate_ranks(results)
        dcg = 0
        idcg = 0
        for index, data in ranks.sort_values('relevant', ascending=False).reset_index().iterrows():
            dcg += (2 ** data['relevant'] - 1) / np.log2(data['rank'] + 1)
            idcg += (2 ** data['relevant'] - 1) / np.log2((index + 1) + 1)
        return dcg / idcg

    def mean_normalized_discounted_cumulative_gain_score(self, results: pd.DataFrame):
        ''' Calculates mean normalized discounted cumulative gain score.
    
        Args:
            results (pd.DataFrame):

        Returns:
            sum / len(qIDs):

        ''' 
        qIDs = results['qID'].unique()
        sum = 0
        for qID in qIDs:
            sum += self.normalized_discounted_cumulative_gain(results[results['qID'] == qID])
        return sum / len(qIDs)

    def mean_reciprocal_rank(self, results: pd.DataFrame):
        ''' Calculates mean reciprocal rank.
    
        Args:
            results (pd.DataFrame):

        Returns:
            sum / len(qIDs):

        ''' 
        qIDs = results['qID'].unique()
        sum = 0

        for qID in qIDs:
            ranks = self.calculate_ranks(results[results['qID'] == qID])
            ranks = ranks.sort_values('rank', ascending=False).iloc[0]
            sum += (1 / ranks['rank'])

        return sum / len(qIDs)
