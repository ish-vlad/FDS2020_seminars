import dask
import dask.dataframe as dd
import dask_ml.metrics
import dask_ml.model_selection
import dask_xgboost
import glob
import os
import pandas as pd
import sklearn
import xgboost

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from utils import Logger, timeit


class RegressionModel:
    def __init__(self, random_seed=42, n_jobs=20, verbose=True):
        # Data fields
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        # Model fields
        self.model = None

        # Train-test split
        self.tts = None
        self.scoring = None

        # Utilities
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logger = Logger(verbose)
        self.random_seed = random_seed
        self.timing, self.accuracy = {}, {}

    def _process_data(self, data, test_size):
        # CLEANING
        data = data[~data['DepDelay'].isnull()].reset_index(drop=True)
        X, y = data, data['DepDelay'] > 0.
        del X['DepDelay']

        # FILLING
        numeric_cols = [
            col for col, dtype in X.dtypes.reset_index().values if dtype == 'int64' or dtype == 'float64'
        ]
        X = X[numeric_cols]
        X.fillna(X.quantile())

        # SPLITTING
        self.X_train, self.X_test, self.y_train, self.y_test = self.tts(X, y, test_size=test_size, shuffle=True,
                                                                   random_state=self.random_seed)

    def _print_train_test_shapes(self):
        print_str = f'Train-test splitting done. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}.'
        self.logger.print_(print_str)

    def load_data(self, data_dir, test_size=0.2):
        raise NotImplementedError('Load data is not implemented')

    @timeit(dictionary_field='timing')
    def optimize(self, param_grid):
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=3, verbose=self.verbose,
                                   scoring=self.scoring)
        grid_search.fit(self.X_train, self.y_train.astype(int))
        self.model = grid_search.best_estimator_

    @timeit(dictionary_field='timing')
    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        raise NotImplementedError('Evaluation is not implemented')

    def test_model(self, data_dir, param_grid):
        self.load_data(data_dir)
        self.optimize(param_grid)
        self.train()
        self.evaluate()

    def __str__(self):
        string = type(self).__name__ + '\n\tTIME.\t'
        string += ', '.join(['%s: %4.2e s' % (k, v) for k, v in self.timing.items()]) + '\n'
        string += '\tROCAUC. Train: %.3f, Test: %.3f' % (self.accuracy['train'], self.accuracy['test'])
        return string


class ScikitModel(RegressionModel):
    def __init__(self, random_seed=42, n_jobs=20, verbose=True):
        super(ScikitModel, self).__init__(random_seed, n_jobs, verbose)

        # Model fields
        self.model = xgboost.XGBClassifier(n_jobs=n_jobs)

        # Train-test split
        self.tts = sklearn.model_selection.train_test_split

    @timeit(dictionary_field='timing')
    def load_data(self, data_dir, test_size=0.2):
        data = pd.concat([
            pd.read_csv(file_path) for file_path in glob.glob(os.path.join(data_dir, '*', '*.csv'))
        ])

        self._process_data(data, test_size)
        self._print_train_test_shapes()

    @timeit(dictionary_field='timing')
    def evaluate(self):
        y_pred_proba = self.model.predict_proba(self.X_train)[:, 1]
        self.accuracy['train'] = roc_auc_score(self.y_train, y_pred_proba)

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        self.accuracy['test'] = roc_auc_score(self.y_test, y_pred_proba)


class DaskModel(RegressionModel):
    def __init__(self, client, random_seed=42, n_jobs=20, verbose=True):
        super(DaskModel, self).__init__(random_seed, n_jobs, verbose)

        # Model fields
        self.model = dask_xgboost.XGBClassifier()
        self.client = client

        self.scoring = DaskModel._acc_score
        self.tts = dask_ml.model_selection.train_test_split

    @staticmethod
    def _acc_score(estimator, X_test, y_true):
        y_pred = estimator.predict(X_test)
        return dask_ml.metrics.accuracy_score(y_true, y_pred)

    @timeit(dictionary_field='timing')
    def load_data(self, data_dir, test_size=0.2):
        data = dd.read_csv(os.path.join(data_dir, '*', '*.csv'), dtype={
            'CRSElapsedTime': 'float64',
            'TailNum': 'object'
        })

        self._process_data(data, test_size)

        self.X_train = self.X_train.to_dask_array(lengths=True)
        self.X_test = self.X_test.to_dask_array(lengths=True)
        self.y_train = self.y_train.to_dask_array(lengths=True)
        self.y_test = self.y_test.to_dask_array(lengths=True)

        self._print_train_test_shapes()

    @timeit(dictionary_field='timing')
    def evaluate(self):
        y_pred_proba = self.client.compute(self.model.predict_proba(self.X_train)).result()
        roc_auc = dask.delayed(roc_auc_score)(self.y_train, y_pred_proba)
        self.accuracy['train'] = roc_auc.compute()

        y_pred_proba = self.client.compute(self.model.predict_proba(self.X_test)).result()
        roc_auc = dask.delayed(roc_auc_score)(self.y_test, y_pred_proba)
        self.accuracy['test'] = roc_auc.compute()
