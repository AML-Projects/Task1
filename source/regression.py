"""
Regression is performd in this class
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import math

import pandas as pd
import xgboost as xgb
from sklearn import model_selection
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVR
# import matplotlib.pyplot as plt
import numpy as np

from logcreator.logcreator import Logcreator


class Regression:
    def __init__(self, name="xgb", stratified_split=True):
        Logcreator.info("Start fit model")
        self.name = name
        self.stratified_split = stratified_split

    def fit_predict(self, x_train, y_train, x_test, handin=False):
        switcher = {
            'ridge': self.ridge_regression,
            'svr': self.svr_regression,
            'xgb': self.xgboost_regression,
            'elastic': self.elastic_net_regression
        }
        reg = switcher.get(self.name)

        return reg(x_train, y_train, x_test, handin)

    def ridge_regression(self, x_train, y_train, x_test, handin):
        x_test_split, x_train_split, y_test_split, y_train_split = self.get_data_split(handin, x_test, x_train, y_train)

        # grid search parameters
        parameters = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            # 'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 1600, 1700, 1799, 1800, 1801, 1805, 1810, 1820, 1850],
            'normalize': [True, False],
            'solver': ['svd', 'saga']
            # 'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }

        # TODO maybe increase nr_folds again?
        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]) / 3)

        model = Ridge(random_state=41)

        ridge, search_results = self.do_grid_search(model, nr_folds, parameters, x_train_split, y_train_split)

        # Use ridge regression
        ridge.fit(x_train_split, y_train_split)
        return ridge, x_test_split, y_test_split, x_train_split, y_train_split, search_results

    def svr_regression(self, x_train, y_train, x_test, handin):
        x_test_split, x_train_split, y_test_split, y_train_split = self.get_data_split(handin, x_test, x_train, y_train)

        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]))

        model = SVR()

        # grid search parameters
        parameters = {
            "C": [40, 20, 10, 1],
            # "gamma": ['scale', 'auto'],
            # "coef0": [0.0, 0.5, 1.0],
            # "kernel": ['rbf', 'poly'],
            "degree": [1],  # of poly kernel
        }

        best_model, search_results = self.do_grid_search(model, nr_folds, parameters, x_train_split, y_train_split)

        best_model.fit(x_train_split, y_train_split)
        return best_model, x_test_split, y_test_split, x_train_split, y_train_split, search_results

    def xgboost_regression(self, x_train, y_train, x_test, handin):
        x_test_split, x_train_split, y_test_split, y_train_split = self.get_data_split(handin, x_test, x_train, y_train)

        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]) / 3)

        y_train_split = y_train_split.astype(int)
        param = {}
        param['booster'] = ['gbtree']
        param['objective'] = ['reg:squarederror']
        # setting max_depth to high results in overfitting
        param['max_depth'] = [2, 3, 4]
        param['min_child_weight'] = [1, 2, 3]
        # subsampling of rows: lower values of subsample can prevent overfitting
        param['subsample'] = [i / 10. for i in range(8, 11)]
        # subsampling of columns:
        # param['colsample_bytree'] = [i / 10. for i in range(8, 11)]
        # learning rate eta
        param['eta'] = [.1, 0.09, 0.08]

        model = xgb.XGBRegressor(random_state=41)

        best_model, search_results = self.do_grid_search(model, nr_folds, param, x_train_split, y_train_split)

        return best_model, x_test_split, y_test_split, x_train_split, y_train_split, search_results

    def elastic_net_regression(self, x_train, y_train, x_test, handin):
        x_test_split, x_train_split, y_test_split, y_train_split = self.get_data_split(handin, x_test, x_train, y_train)

        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]) / 3)

        model = ElasticNet(random_state=43)

        # grid search parameters
        parameters = {
            'alpha': [0.2, 0.5, 1, 10, 100],
            'l1_ratio': [0.25, 0.5, 0.75],
            'normalize': [False]
        }
        best_model, search_results = self.do_grid_search(model, nr_folds, parameters, x_train_split, y_train_split)

        best_model.fit(x_train_split, y_train_split)
        return best_model, x_test_split, y_test_split, x_train_split, y_train_split, search_results

    def do_grid_search(self, model, nr_folds, parameters, x_train_split, y_train_split):
        # TODO Maybe use Stratified split
        # cv = RepeatedStratifiedKFold(n_splits=nr_folds, n_repeats=1, random_state=1)
        grid_search = model_selection.GridSearchCV(model, parameters,
                                                   # scoring='neg_mean_squared_error',
                                                   scoring='r2',
                                                   # use every cpu thread
                                                   n_jobs=-1,
                                                   # Refit an estimator using the best found parameters
                                                   refit=True,
                                                   cv=nr_folds,
                                                   # Return train score to check for overfitting
                                                   return_train_score=True,
                                                   verbose=1)
        grid_search.fit(x_train_split, y_train_split)

        # Best estimator
        Logcreator.info("Best estimator from GridSearch: {}".format(grid_search.best_estimator_))
        Logcreator.info("Best alpha found: {}".format(grid_search.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(grid_search.best_score_))

        # make pandas print everything
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        results = pd.DataFrame(grid_search.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)

        Logcreator.info(
            results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']].head(30))

        best_model = grid_search.best_estimator_

        return best_model, results

    def strat_split(self, x_train, y_train, nrOfBuckets):
        x_train_strat = x_train
        y_train_strat = y_train
        # plt.figure()
        # y_train.plot.hist()
        # plt.show()
        """
        Fill samples into buckets to generate a certain amount of classes for the stratified split
        """

        bucketLabels = [a for a in range(nrOfBuckets)]
        try:
            #y_train_strat = y_train_strat.rename(columns={0: 'y'})
            y_train_strat['buckets'] = pd.qcut(y_train[0], q=nrOfBuckets, labels=bucketLabels)

            Logcreator.info("Nr of buckets for stratified split: " + str(nrOfBuckets))
        except ValueError:
            x_train_strat, y_train_strat, train_data_x_removed, train_data_y_removed = self.strat_split(x_train, y_train, nrOfBuckets-1)
            return x_train_strat, y_train_strat, train_data_x_removed, train_data_y_removed
        y_train_strat['buckets_int'] = y_train_strat['buckets'].astype(int)
        y_train_strat = y_train_strat.drop(['buckets'], axis=1)

        """
         Remove samples for which we have only one age class, 
         so that we can do a stratified split
        """
        # y_train_strat['buckets_int'] = y_train_strat['buckets'].astype(int)
        counts = y_train_strat['buckets_int'].value_counts()
        Logcreator.info("The number of samples for each 'age/class' are:\n", pd.DataFrame(counts).T)

        classes_with_one_el = counts[counts == 1].index.values
        Logcreator.info("'Age' with only one sample:", classes_with_one_el)

        remove_columns = y_train_strat["buckets_int"].isin(classes_with_one_el)

        # Save samples/rows we remove
        train_data_x_removed = x_train_strat[remove_columns]
        train_data_y_removed = y_train_strat[remove_columns]

        y_train_strat = y_train_strat[~remove_columns]
        x_train_strat = x_train_strat[~remove_columns]

        size_testset = x_train_strat.shape[0] * 0.1
        nrClasses = y_train['buckets_int'].nunique()
        if size_testset < nrClasses:
            x_train_strat, y_train_strat, train_data_x_removed, train_data_y_removed = self.strat_split(x_train, y_train, size_testset)
            return x_train_strat, y_train_strat, train_data_x_removed, train_data_y_removed
        return x_train_strat, y_train_strat, train_data_x_removed, train_data_y_removed


    def get_data_split(self, handin, x_test, x_train, y_train):
        if not handin:
            if self.stratified_split:
                Logcreator.info("Stratified split")
                # convert to dataframe if they are not already
                x_train = pd.DataFrame(x_train)
                y_train = pd.DataFrame(y_train)
                x_test = pd.DataFrame(x_test)

                # check if the indexes of x_train and y_train match
                if ((x_train.index == y_train.index).all() == False):
                    Logcreator.warn("Indexes do not match, probably because of outlier removal!")
                    # drop indices because they are different
                    x_train.reset_index(drop=True, inplace=True)
                    y_train.reset_index(drop=True, inplace=True)


                nrBuckets = 30
                x_train, y_train, train_data_x_removed, train_data_y_removed = self.strat_split(x_train, y_train, nrBuckets)
                # Do the stratified split without those samples
                x_train_split, x_test_split, y_train_split, y_test_split = model_selection.train_test_split(
                    x_train, y_train, stratify=y_train['buckets_int'], test_size=0.1, random_state=43)

                # Append the removed samples/rows to the training split
                x_train_split = x_train_split.append(train_data_x_removed)
                y_train_split = y_train_split.append(train_data_y_removed)
                y_train_split = y_train_split.drop(['buckets_int'], axis=1)
                y_test_split = y_test_split.drop(['buckets_int'], axis=1)

                # TODO maybe move this code to where we need it
                # reset all indexes(=start from 0) because some transformers remove the index
                # and because y is not part of every transformation, the index is not removed for y
                # which can causes problems
                x_train_split.reset_index(drop=True, inplace=True)
                x_test_split.reset_index(drop=True, inplace=True)
                y_train_split.reset_index(drop=True, inplace=True)
                y_test_split.reset_index(drop=True, inplace=True)

            else:
                x_train_split, x_test_split, y_train_split, y_test_split = model_selection.train_test_split(
                    x_train, y_train, test_size=0.1, random_state=43)
        else:
            x_train_split = x_train
            x_test_split = x_test
            y_train_split = y_train
            y_test_split = None

        Logcreator.info("x_train_split: {}".format(x_train_split.shape))
        Logcreator.info("x_test_split: {}".format(x_test_split.shape))

        y_train_split = y_train_split.values.ravel()

        return x_test_split, x_train_split, y_test_split, y_train_split
