"""
Data normalizer
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from logcreator.logcreator import Logcreator


class Normalizer:
    def __init__(self, name, fit_on):
        self.name = name
        self.fit_on = fit_on
        Logcreator.info("Start normalizer")

    def transform_custom(self, x_train, y_train, x_test):
        switcher = {
            'stdscaler': self.standard_scaler,
            'minmaxscaler': self.minmax_scaler,
            'robustscaler': self.robust_scaler
        }
        norm = switcher.get(self.name)

        return norm(x_train=x_train, y_train=y_train, x_test=x_test)

    def to_DataFrame(self, x_train, y_train, x_test):
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(y_train)
        x_test = pd.DataFrame(x_test)
        return x_train, y_train, x_test

    def standard_scaler(self, x_train, y_train, x_test):
        return self.scale_data(StandardScaler(), x_train, y_train, x_test)

    def minmax_scaler(self, x_train, y_train, x_test):
        return self.scale_data(MinMaxScaler(), x_train, y_train, x_test)

    def robust_scaler(self, x_train, y_train, x_test):
        return self.scale_data(RobustScaler(), x_train, y_train, x_test)

    def scale_data(self, scaler, x_train, y_train, x_test):
        x_train, y_train, x_test = self.to_DataFrame(x_train, y_train, x_test)
        if self.fit_on == 'test':
            scaler = scaler.fit(x_test)
        elif self.fit_on == 'both':
            scaler = scaler.fit(x_train.append(x_test))
        else:
            scaler = scaler.fit(x_train)

        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, y_train, x_test_scaled
