"""
Data imputer to set values for NaN values in datasets
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer

from logcreator.logcreator import Logcreator


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, name='median', knn_weights='distance', knn_n_neighbors=20, iterative_n_nearest_features=10):
        Logcreator.info("Imputer initialized")
        self.name = name
        self.knn_weights = knn_weights
        self.knn_n_neighbors = knn_n_neighbors
        self.iterative_n_nearest_features = iterative_n_nearest_features

    # TODO if we want to use sklearn grid_search and pipeline we would need to implement fit and transform
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def transform_custom(self, x_train, y_train, x_test):
        switcher = {
            'mean': self.mean_simple_imputer,
            'median': self.median_simple_imputer,
            'iterative': self.multivariate_imputer,
            'knn': self.knn_imputer
        }
        imp = switcher.get(self.name)
        return imp(x_train=x_train, y_train=y_train, x_test=x_test)

    def mean_simple_imputer(self, x_train, y_train, x_test):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(x_train)
        x_train_imputed = imp.transform(x_train)
        x_test_imputed = imp.transform(x_test)
        Logcreator.info("Imputation result with mean imputation for x_train: \n")
        Logcreator.info(pd.DataFrame(x_train_imputed).head())
        Logcreator.info("Imputation result with mean imputation for x_test: \n")
        Logcreator.info(pd.DataFrame(x_test_imputed).head())
        return x_train_imputed, y_train, x_test_imputed

    def median_simple_imputer(self, x_train, y_train, x_test):
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.fit(x_train)
        x_train_imputed = imp.transform(x_train)
        x_test_imputed = imp.transform(x_test)
        Logcreator.info("Imputation result with median imputation for x_train: \n")
        Logcreator.info(pd.DataFrame(x_train_imputed).head())
        Logcreator.info("Imputation result with median imputation for x_test: \n")
        Logcreator.info(pd.DataFrame(x_test_imputed).head())
        return x_train_imputed, y_train, x_test_imputed

    def multivariate_imputer(self, x_train, y_train, x_test):
        Logcreator.info("Nr. of rows without any NaN: {} \n".format(
            x_train.shape[0] - sum([True for idx, row in x_train.iterrows() if any(row.isna())])))

        imp = IterativeImputer(missing_values=np.nan, max_iter=10,
                               sample_posterior=True,
                               initial_strategy='median',
                               # Nearness between features is measured using the absolute correlation coefficient between each feature pair
                               n_nearest_features=self.iterative_n_nearest_features,
                               random_state=0)
        imp.fit(x_train)
        Logcreator.info(x_train.head())

        x_train_imputed = imp.transform(x_train)
        x_test_imputed = imp.transform(x_test)
        return x_train_imputed, y_train, x_test_imputed

    def knn_imputer(self, x_train, y_train, x_test):
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=self.knn_n_neighbors, weights=self.knn_weights)

        x_train_imputed = imputer.fit_transform(x_train)
        x_test_imputed = imputer.transform(x_test)

        return x_train_imputed, y_train, x_test_imputed
