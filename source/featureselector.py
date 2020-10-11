"""
Data normalizer
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, SelectFromModel

from logcreator.logcreator import Logcreator


class TreeBasedFeatureSelector(TransformerMixin):
    """
    Selecting features base on impurity.
    """

    def __init__(self, n_estimators=100):
        self.clf = ExtraTreesClassifier(criterion='gini',
                                        n_estimators=n_estimators,
                                        bootstrap=False,
                                        # class_weight='balanced',
                                        random_state=41)

    def fit(self, X, y):
        """
        Expecting y to be a pandas DataFrame.
        """
        self.clf = self.clf.fit(X, y.values.ravel())
        return self

    def transform(self, X):
        self.model = SelectFromModel(self.clf, prefit=True)
        x_transformed = self.model.transform(X)
        return x_transformed


class FeatureSelector:
    """
    Partly based on # https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
    """

    def __init__(self, k=200, corr_threshold=0.8):
        Logcreator.info("Feature Selection")
        self.k = k
        self.correlation_threshold = corr_threshold

    def remove_constant_features(self, x_train, y_train, x_test):
        """
        Remove features with zero or almost zero variance depending on the threshold.
        """
        constant_filter = VarianceThreshold(threshold=0)

        x_train_feature = constant_filter.fit_transform(x_train, y_train)
        x_test_feature = constant_filter.transform(x_test)

        Logcreator.info("Variance Threshold ", constant_filter.get_params()['threshold'],
                        " -> nr of features: ",
                        x_train_feature.shape[1])

        return x_train_feature, y_train, x_test_feature

    def remove_correlated_features(self, x_train, y_train, x_test):
        """
        Removes feature based on how much they correlate.
        """
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)

        correlated_features = set()
        correlation_matrix = pd.DataFrame(x_train).corr()

        self.correlation_threshold = 0.8

        # TODO maybe don't remove all correlated features, but kee one of them
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_threshold:
                    column_name = correlation_matrix.columns[i]
                    correlated_features.add(column_name)

        x_train.drop(labels=correlated_features, axis=1, inplace=True)
        x_test.drop(labels=correlated_features, axis=1, inplace=True)

        Logcreator.info("Correlation Threshold ", self.correlation_threshold,
                        " -> nr of features: ",
                        x_train.shape[1])

        return x_train, y_train, x_test

    def selectBestK(self, x_train, y_train, x_test):
        """
        Expecting y to be a pandas Dataframe.
        """

        # score_func: f_regression, mutual_info_regression, ...
        k_best = SelectKBest(score_func=mutual_info_regression, k=self.k)

        x_train_best = k_best.fit_transform(x_train, y_train.values.ravel())
        x_test_best = k_best.transform(x_test)

        Logcreator.info("SelectKBest k =", self.k, " -> nr features: ", str(x_train_best.shape[1]))

        return x_train_best, y_train, x_test_best

    def selectBestBasedOnImpurity(self, x_train, y_train, x_test):
        selector = TreeBasedFeatureSelector()

        x_train_best = selector.fit_transform(x_train, y_train)
        x_test_best = selector.transform(x_test)

        Logcreator.info("TreeBasedFeatureSelector", " -> nr features: ", str(x_train_best.shape[1]))

        return x_train_best, y_train, x_test_best
