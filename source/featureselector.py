"""
Feature selection
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

    def __init__(self, remove_constant=True,
                 remove_constant_threshold=0.01,
                 remove_correlated=True,
                 remove_correlated_threshold=0.8,
                 use_select_best_k=True,
                 k=200,
                 use_select_best_based_on_impurity=False,
                 ):
        Logcreator.info("Feature Selection")

        if isinstance(remove_constant, str):
            remove_constant = remove_constant == "True"
        if isinstance(remove_correlated, str):
            remove_correlated = remove_correlated == "True"
        if isinstance(use_select_best_k, str):
            use_select_best_k = use_select_best_k == "True"
        if isinstance(use_select_best_based_on_impurity, str):
            use_select_best_based_on_impurity = use_select_best_based_on_impurity == "True"

        self.remove_constant = remove_constant
        self.constant_features_threshold = remove_constant_threshold
        self.remove_correlated = remove_correlated
        self.correlation_threshold = remove_correlated_threshold
        self.use_select_best_k = use_select_best_k
        self.k = k
        self.use_selectBestBasedOnImpurity = use_select_best_based_on_impurity

    def transform_custom(self, x_train, y_train, x_test):
        if self.remove_constant:
            x_train, y_train, x_test = self.remove_constant_features(x_train,
                                                                     y_train,
                                                                     x_test,
                                                                     0.0)

            x_train, y_train, x_test = self.remove_constant_features(x_train,
                                                                     y_train,
                                                                     x_test,
                                                                     self.constant_features_threshold)

        # We do not remove duplicates as there are no duplicate features in the dataset
        fs_remove_duplicate = False
        if fs_remove_duplicate:
            x_train, y_train, x_test = self.remove_duplicates(x_train,
                                                              y_train,
                                                              x_test)

        if self.remove_correlated:
            x_train, y_train, x_test = self.remove_correlated_features(x_train,
                                                                       y_train,
                                                                       x_test)

        if self.use_select_best_k:
            x_train, y_train, x_test = self.selectBestK(x_train,
                                                        y_train,
                                                        x_test)

        if self.use_selectBestBasedOnImpurity:
            x_train, y_train, x_test = self.selectBestBasedOnImpurity(x_train,
                                                                      y_train,
                                                                      x_test)

        return x_train, y_train, x_test

    def remove_features_with_many_Nan(self, x_train, y_train, x_test):
        """
        Remove features which have a certain amount of nan values
        """
        nrRows = x_train.shape[0]
        nrCols = x_train.shape[1]
        threshold = 0.1
        featuresToRemove = set()
        for column in x_train.columns:
            sum = x_train[column].isnull().sum()
            if sum / nrRows > threshold:
                featuresToRemove.add(column)
        x_train.drop(labels=featuresToRemove, axis=1, inplace=True)
        x_test.drop(labels=featuresToRemove, axis=1, inplace=True)
        return x_train, y_train, x_test

    def remove_constant_features(self, x_train, y_train, x_test, threshold=0.0):
        """
        Remove features with zero or almost zero variance depending on the threshold.
        """
        Logcreator.info("\nRemove constant features:")
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)

        constant_filter = VarianceThreshold(threshold=threshold)

        constant_filter.fit(x_train, y_train)

        constant_columns = [column for column in x_train.columns
                            if column not in x_train.columns[constant_filter.get_support()]]
        constColumns = ""
        for column in constant_columns:
            constColumns += (str(column) + ", ")
        Logcreator.info("Following columns are constant: " + str(constant_columns))

        x_train_feature = constant_filter.transform(x_train)
        x_test_feature = constant_filter.transform(x_test)

        Logcreator.info("Variance Threshold ", constant_filter.get_params()['threshold'],
                        " -> nr of features: ",
                        x_train_feature.shape[1])
        Logcreator.info("Removed: " + str(x_train.shape[1] - x_train_feature.shape[1]) + " features")

        return x_train_feature, y_train, x_test_feature

    def remove_duplicates(self, x_train, y_train, x_test):
        """
        Detect duplicate columns in training set and remove those columns in training and testset
        """
        Logcreator.info("\nRemove Duplicates:")
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)

        train_features_T = x_train.T
        duplicateFeatures = train_features_T.duplicated()
        print(duplicateFeatures.sum())  # Is always 0 for our dataset, therefore we don't do duplicate removing

    def remove_correlated_features(self, x_train, y_train, x_test):
        """
        Removes feature based on how much they correlate.
        """
        Logcreator.info("\nRemove Correlated Features:")
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)

        correlated_features = set()
        correlation_matrix = pd.DataFrame(x_train).corr()

        self.correlation_threshold = 0.8

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_threshold:
                    column_name = correlation_matrix.columns[i]
                    correlated_features.add(column_name)

        x_train.drop(labels=correlated_features, axis=1, inplace=True)
        x_test.drop(labels=correlated_features, axis=1, inplace=True)
        Logcreator.info("Following features are removed: " + str(correlated_features))
        Logcreator.info("Nr. of features remvoed: " + str(len(correlated_features)))
        Logcreator.info("Correlation Threshold ", self.correlation_threshold,
                        " -> nr of features: ",
                        x_train.shape[1])

        return x_train, y_train, x_test

    def selectBestK(self, x_train, y_train, x_test):
        """
        Expecting y to be a pandas Dataframe.
        """
        Logcreator.info("\nSelect Best k features:")
        # score_func: f_regression, mutual_info_regression, ...
        k_best = SelectKBest(score_func=mutual_info_regression, k=self.k)

        # Not currently used, here the mutual information of each feature is calculated, k could be choosen according to that result
        if False:  # TODO throws error
            mi = mutual_info_regression(x_train, y_train)
            mi = pd.Series(mi)
            mi.index = x_train.columns
            sum = 0
            for feature in mi:
                if feature == 0:
                    sum += 1
            nr_of_relevant_features = mi.shape[1] - sum  # Could be us as k

        k_best = SelectKBest(score_func=mutual_info_regression, k=self.k)
        x_train_best = k_best.fit_transform(x_train, y_train.values.ravel())
        x_test_best = k_best.transform(x_test)

        Logcreator.info("SelectKBest k =", self.k, " -> nr features: ", str(x_train_best.shape[1]))

        return x_train_best, y_train, x_test_best

    def selectBestBasedOnImpurity(self, x_train, y_train, x_test):
        Logcreator.info("\nSelect Best Based on Impurity")
        selector = TreeBasedFeatureSelector()

        x_train_best = selector.fit_transform(x_train, y_train)
        x_test_best = selector.transform(x_test)

        Logcreator.info("TreeBasedFeatureSelector", " -> nr features: ", str(x_train_best.shape[1]))

        return x_train_best, y_train, x_test_best
