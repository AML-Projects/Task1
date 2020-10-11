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
                                pd.DataFrame(counts).T.reset_index())

                classes_with_one_el = counts[counts == 1].index.values
                Logcreator.info("'Age' with only one sample:\n", classes_with_one_el)

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

        # Run Gridsearch for alpha
        parameters = {
            'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 1600,
                      1700, 1799, 1800, 1801, 1805, 1810, 1820, 1850]}
        nr_folds = math.floor(math.sqrt(x_train_split.shape[0]))
        model = Ridge()
        ridge_regressor = model_selection.GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=nr_folds)
        ridge_regressor.fit(x_train_split, y_train_split)

        # Best estimator
        Logcreator.info("Best estimator from GridSearch: {}".format(ridge_regressor.best_estimator_))
        Logcreator.info("Best alpha found: {}".format(ridge_regressor.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(ridge_regressor.best_score_))

        ridge = ridge_regressor.best_estimator_

        # Use ridge regression
        ridge.fit(x_train_split, y_train_split)
        return ridge, x_test_split, y_test_split, x_train_split, y_train_split
