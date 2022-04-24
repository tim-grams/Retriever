import pandas as pd
from sklearn.preprocessing import StandardScaler


class Evaluation(object):

    def __init__(self,
                 X_y_train: pd.DataFrame,
                 qrels: pd.DataFrame,
                 accuracy_k: int = 1,
                 precision_k: int = 1):
        self.X_y_train = X_y_train
        self.qrels = qrels
        self.accuracy_k = accuracy_k
        self.precision_k = precision_k
        self.results = pd.DataFrame(columns=['Run',
                                             f'Accuracy@{self.accuracy_k}',
                                             f'Precision@{self.precision_k}'])

        self.y = self.X_y_train['y']
        X = self.X_y_train.drop(columns=['qID', 'pID', 'y'])
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X)

    def __call__(self, *args, **kwargs):
        pass

    def compute_metrics(self, model):
        model.fit(self.X, self.y)

        print(f'MAP:')

    def MAP_score(self, results: pd.DataFrame):
        results.groupby('qID')['confidence'].rank(method='average', ascending=False)
        ranks = results[results['relevant'] >= 1].reset_index()
        sum_inverse = 0
        for i in range(0, len(ranks)):
            sum_inverse += 1 / ranks['rank'][i]
        MAP = 1 / len(ranks) * sum_inverse
        return MAP

    def nDCG(self, results: pd.DataFrame):
        pass

    def precision(self, k):
        pass

    def recall(self, k):
        pass
