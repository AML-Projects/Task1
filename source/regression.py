"""
Regression is performd in this class
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import math

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import Ridge

from logcreator.logcreator import Logcreator


class Regression:
    def __init__(self, stratified_split=True):
        self.stratified_split = stratified_split
        Logcreator.info("Start fit model")

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

        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]))

        model = Ridge(random_state=41)

        ridge = self.do_grid_search(model, nr_folds, parameters, x_train_split, y_train_split)

        # Use ridge regression
        ridge.fit(x_train_split, y_train_split)
        return ridge, x_test_split, y_test_split, x_train_split, y_train_split

    def do_grid_search(self, model, nr_folds, parameters, x_train_split, y_train_split):
        grid_search = model_selection.GridSearchCV(model, parameters,
                                                   # scoring='neg_mean_squared_error',
                                                   scoring='r2',
                                                   # use every cpu thread
                                                   n_jobs=-1,
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

        ridge = grid_search.best_estimator_

        return ridge

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

                """
                 Remove samples for which we have only one age class, 
                 so that we can do a stratified split
                """
                y_train_int = y_train.astype(int)
                counts = y_train_int["y"].value_counts()
                Logcreator.info("The number of samples for each 'age/class' are:\n",
                                pd.DataFrame(counts).T)

                classes_with_one_el = counts[counts == 1].index.values
                Logcreator.info("'Age' with only one sample:", classes_with_one_el)

                remove_columns = y_train_int["y"].isin(classes_with_one_el)

                # Save samples/rows we remove
                train_data_x_removed = x_train[remove_columns]
                train_data_y_removed = y_train[remove_columns]

                # Remove samples/rows
                y_train_int = y_train_int[~remove_columns]
                y_train = y_train[~remove_columns]
                x_train = x_train[~remove_columns]

                # Do the stratified split without those samples
                x_train_split, x_test_split, y_train_split, y_test_split = model_selection.train_test_split(
                    x_train, y_train, stratify=y_train_int, test_size=0.1, random_state=43)

                # Append the removed samples/rows to the training split
                x_train_split = x_train_split.append(train_data_x_removed)
                y_train_split = y_train_split.append(train_data_y_removed)

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

        return x_test_split, x_train_split, y_test_split, y_train_split
