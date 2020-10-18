"""
Feature selection
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler

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

    def __init__(self, remove_constant,
                 remove_constant_threshold,
                 remove_correlated,
                 remove_correlated_threshold,
                 use_select_best_k,
                 k,
                 use_select_best_based_on_impurity,
                 lasso_alpha=0.2,
                 lasso_on=False,
                 lda_on=False,
                 lda_n_components=None,
                 pca_explained_var_threshold=0.99  # explained_var between 0 and 1
                 ):
        Logcreator.info("Feature Selection")

        if isinstance(remove_constant, str):
            remove_constant = remove_constant == "True"
        if isinstance(remove_correlated, str):
            remove_correlated = remove_correlated == "True"
        if isinstance(use_select_best_k, str):
            use_select_best_k = use_select_best_k == "True"
        if isinstance(lasso_on, str):
            lasso_on = lasso_on == "True"
        if isinstance(lda_on, str):
            use_lda = lda_on == "True"
        if isinstance(lda_n_components, str):
            if lda_n_components == "None":
                lda_n_components = None
        if isinstance(use_select_best_based_on_impurity, str):
            use_select_best_based_on_impurity = use_select_best_based_on_impurity == "True"

        self.pca_explained_var_threshold = pca_explained_var_threshold
        self.remove_constant = remove_constant
        self.constant_features_threshold = remove_constant_threshold
        self.remove_correlated = remove_correlated
        self.correlation_threshold = remove_correlated_threshold
        self.use_select_best_k = use_select_best_k
        self.k = k
        self.lasso_alpha = lasso_alpha
        self.lasso_on = lasso_on
        self.lda_on = lda_on
        self.lda_n_components = lda_n_components
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
        if self.lda_on:
            x_train, y_train, x_test = self.LDA(x_train, y_train, x_test)

        if self.remove_correlated:
            x_train, y_train, x_test = self.remove_correlated_features(x_train,
                                                                       y_train,
                                                                       x_test)
        fs_do_pca = False
        if fs_do_pca:
            # gives very bad results
            x_train, y_train, x_test = self.selectTransformWithPCA(x_train, y_train, x_test)

        if self.lasso_on:
            x_train, y_train, x_test = self.selectedBasedOnLasso(x_train, y_train, x_test)

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

        constant_filter.fit(x_train)

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

    def selectedBasedOnLasso(self, x_train, y_train, x_test):
        Logcreator.info("\nLasso:")
        # scaling because we fit an estimator
        rs = StandardScaler()
        x_train = rs.fit_transform(x_train)
        x_test = rs.transform(x_test)

        estimator = Lasso(alpha=self.lasso_alpha, max_iter=10e4, random_state=41)
        # estimator = ElasticNet(alpha=self.lasso_alpha, max_iter=10e4, l1_ratio=0.8, random_state=41)

        estimator.fit(x_train, y_train)
        selector = SelectFromModel(estimator=estimator)  # , max_features=80, threshold=-np.inf)

        x_train_lasso = selector.fit_transform(x_train, y_train)
        x_test_lasso = selector.transform(x_test)

        Logcreator.info("LASSO alpha:", self.lasso_alpha,
                        " -> removed: " + str(x_train.shape[1] - x_train_lasso.shape[1]) + " features")
        Logcreator.info("LASSO", " -> nr features: ", str(x_train_lasso.shape[1]))

        return x_train_lasso, y_train, x_test_lasso

    def LDA(self, x_train, y_train, x_test):
        Logcreator.info("\nLDA")
        clf = LinearDiscriminantAnalysis(n_components=self.lda_n_components)

        clf.fit(x_train, y_train)
        x_train_lda = clf.transform(x_train)
        x_test_lda = clf.transform(x_test)

        Logcreator.info("LDA removed: " + str(x_train.shape[1] - x_train_lda.shape[1]) + " features")
        Logcreator.info("LDA", " -> nr of features: ", x_train_lda.shape[1])

        return x_train_lda, y_train, x_test_lda


def selectTransformWithPCA(self, x_train, y_train, x_test):
    Logcreator.info("\nSelect based on PCA:")
    # we have to scale the data first
    # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
    rs = RobustScaler()
    x_train = rs.fit_transform(x_train)
    x_test = rs.transform(x_test)

    pca = PCA(n_components=self.pca_explained_var_threshold, svd_solver='auto')
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    explained_variance = pca.explained_variance_ratio_
    # print("PCA explained variance", explained_variance)

    nr_components = 2
    principal_components = np.transpose(x_train_pca)
    for i in range(0, nr_components):
        fig, ax = plt.subplots()
        ax.scatter(principal_components[i], y_train)
        ax.set(xlabel='pc' + str(i + 1), ylabel='age', title='principle component ' + str(i + 1) + ' vs age')
        ax.grid()
        plt.show()

    Logcreator.info("PCA removed: " + str(x_train.shape[1] - x_train_pca.shape[1]) + " features")

    Logcreator.info("PCA ", " -> nr of features: ", x_train_pca.shape[1])

    return x_train_pca, y_train, x_test_pca
