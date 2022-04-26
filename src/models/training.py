import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


class Evaluation(object):

    def __init__(self,
                 X_y_train: pd.DataFrame,
                 X_y_test: pd.DataFrame,
                 qrels: pd.DataFrame,
                 accuracy_k: int = 1,
                 precision_k: int = 1):
        self.X_y_train = X_y_train
        self.X_y_test = X_y_test
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

        self.pair = self.X_y_test['pID', 'qID']
        X_test = self.X_y_test.drop(columns=['pID', 'qID'])
        self.X_test = scaler.transform(X_test)

    def __call__(self, *args, **kwargs):
        pass

    def compute_metrics(self):
        model = GaussianNB()
        model.fit(self.X, self.y)
        results = pd.DataFrame({
            'confidence': model.predict_proba(self.X_test),
            'qID': self.pair['qID'],
            'pID': self.pair['pID']
        })

        print(f'MAP:{self.MAP_score(results)}')

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
