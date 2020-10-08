"""
Outlier detection
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

from logcreator.logcreator import Logcreator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


class CustomOutlierRemover:
    """
    Seems to remove a lot of rows with default parameters for our problem.
    Works with NaN values in data!
    Based on: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
    """

    def __init__(self, strategy='z_score', threshold=None, verbose=0):
        """
        Remove outliers according to strategy.
        :param strategy: ['z_score', 'iqr' ]
        :param threshold: default 3 for z_score, 1.5 for iqr
        :param verbose:
        """
        self.strategy = strategy
        self.threshold = threshold
        if self.threshold is None:
            if self.strategy == 'z_score':
                self.threshold = 3
            else:
                self.threshold = 1.5
        self.verbose = verbose

    def remove_z_score_outliers(self, x, y=None, threshold=3.0):
        x_df = pd.DataFrame(x)

        ''' with division
        z = np.abs(stats.zscore(x_df, nan_policy='omit'))
        # works with NaN because comparing with NaN results always in False
        remove_rows = (z >= threshold).any(axis=1)
        '''

        # without division
        remove_rows = (np.abs(x_df - self.z_score_mean) > (threshold * self.z_score_std)).any(axis=1)

        return self.remove_outliers(remove_rows, x_df, y)

    def remove_iqr_outliers(self, x, y=None, threshold=1.5):
        x_df = pd.DataFrame(x)
        iqr = self.quartile_3 - self.quartile_1
        lower_bound = self.quartile_1 - (iqr * threshold)
        upper_bound = self.quartile_3 + (iqr * threshold)
        # works with NaN because comparing with NaN results always in False
        remove_rows = ((x_df < lower_bound) | (x_df > upper_bound)).any(axis=1)

        return self.remove_outliers(remove_rows, x_df, y)

    def remove_outliers(self, remove_rows, x_df, y):
        if self.verbose == 1:
            nr_removed = remove_rows[remove_rows == True]
            Logcreator.info("OutlierRemover: Removed " + str(nr_removed.shape) + " rows")

        x_out_df = x_df[~remove_rows]
        if y is not None:
            y_df = pd.DataFrame(y)
            y_out_df = y_df[~remove_rows]
            return x_out_df, y_out_df
        else:
            return x_out_df

    def fit(self, x, y=None):
        x_df = pd.DataFrame(x)
        if self.strategy == 'z_score':
            self.z_score_mean = x_df.mean(skipna=True)
            self.z_score_std = x_df.std(skipna=True)
        else:
            self.quartile_1 = x_df.quantile(0.25, axis='index')
            self.quartile_3 = x_df.quantile(0.75, axis='index')

        return self

    def remove(self, x, y=None):
        if self.strategy == 'z_score':
            return self.remove_z_score_outliers(x, y, self.threshold)
        else:
            return self.remove_iqr_outliers(x, y, self.threshold)


class Outliers:
    def __init__(self):
        Logcreator.info("Start outlier detection")

    def LOF(self, x_train, y_train, x_test):
        lof = LocalOutlierFactor(contamination='auto')
        outliers = lof.fit_predict(x_train)
        mask = outliers != -1
        x_train_outl, y_train_outl = x_train[mask, :], y_train[mask]
        Logcreator.info("Nr. of outliers removed: {}".format(x_train.shape[0] - x_train_outl.shape[0]))
        return x_train_outl, y_train_outl, x_test

    def iForest(self, x_train, y_train, x_test):
        ifo = IsolationForest(contamination='auto')
        outliers = ifo.fit_predict(x_train)
        mask = outliers != -1
        x_train_outl, y_train_outl = x_train[mask, :], y_train[mask]
        Logcreator.info("Nr. of outliers removed: {}".format(x_train.shape[0] - x_train_outl.shape[0]))
        return x_train_outl, y_train_outl, x_test

    def customOR(self, x_train, y_train, x_test):
        # strategy: z_score or iqr
        cor = CustomOutlierRemover(strategy='iqr', threshold=10, verbose=1)
        cor.fit(x_train)
        x_train_outl, y_train_outl = cor.remove(x_train, y_train)

        return x_train_outl, y_train_outl, x_test
