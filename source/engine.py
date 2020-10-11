"""
Runs trainings and predictions
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import pandas as pd
from sklearn.metrics import r2_score

from helpers import argumenthelper
from logcreator.logcreator import Logcreator
from source.configuration import Configuration
from source.featureselector import FeatureSelector
from source.imputers import Imputer
from source.normalizer import Normalizer
from source.outliers import Outliers
from source.regression import Regression


class Engine:
    def __init__(self):
        Logcreator.info("Training initialized")

    def train(self, x_train, y_train, x_test):

        # Imputer
        imputer = Imputer()
        imputer_type = Configuration.get('imputer.name')
        switcher = {
            'mean': imputer.mean_simple_imputer,
            'median': imputer.median_simple_imputer,
            'iterative': imputer.multivariate_imputer
        }
        imp = switcher.get(imputer_type)
        x_train_imp, y_train_imp, x_test_imp = imp(x_train=x_train, y_train=y_train, x_test=x_test)

        # Outliers
        outliers = Outliers()
        outliers_method = Configuration.get('outliers.name')
        switcher = {
            'lof': outliers.LOF,
            'iforest': outliers.iForest
        }
        outl = switcher.get(outliers_method)
        x_train_outl, y_train_outl, x_test_outl = outl(x_train=x_train_imp, y_train=y_train_imp, x_test=x_test_imp)

        # Feature selection
        feature_selector = FeatureSelector()
        x_train_fs = pd.DataFrame(x_train_outl)
        y_train_fs = pd.DataFrame(y_train_outl)
        x_test_fs = pd.DataFrame(x_test_outl)
        fs_remove_constant = Configuration.get('feature_selector.remove_constant_features')
        if (fs_remove_constant):
            x_train_fs, y_train_fs, x_test_fs = feature_selector.remove_constant_features(x_train_fs,
                                                                                          y_train_fs,
                                                                                          x_test_fs)
        fs_remove_correlated = Configuration.get('feature_selector.remove_correlated_features')
        if (fs_remove_correlated):
            x_train_fs, y_train_fs, x_test_fs = feature_selector.remove_correlated_features(x_train_fs,
                                                                                            y_train_fs,
                                                                                            x_test_fs)
        fs_selectBestK = Configuration.get('feature_selector.selectBestK')
        if (fs_selectBestK):
            x_train_fs, y_train_fs, x_test_fs = feature_selector.selectBestK(x_train_fs,
                                                                             y_train_fs,
                                                                             x_test_fs)
        fs_selectBestBasedOnImpurity = Configuration.get('feature_selector.selectBestBasedOnImpurity')
        if (fs_selectBestBasedOnImpurity):
            x_train_fs, y_train_fs, x_test_fs = feature_selector.selectBestBasedOnImpurity(x_train_fs,
                                                                                           y_train_fs,
                                                                                           x_test_fs)

        # Normalizer
        normalizer = Normalizer()
        normalizer_method = Configuration.get('normalizer.name')
        switcher = {
            'stdscaler': normalizer.standard_scaler,
            'minmaxscaler': normalizer.minmax_scaler,
            'robustscaler': normalizer.robust_scaler
        }
        norm = switcher.get(normalizer_method)
        x_train_norm, y_train_norm, x_test_norm = norm(x_train=x_train_outl, y_train=y_train_outl, x_test=x_test_outl)

        # Regression
        regression = Regression()
        regression_method = Configuration.get('regression.name')
        switcher = {
            'ridge': regression.ridge_regression
        }
        reg = switcher.get(regression_method)
        regressor, x_test_split, y_test_split, x_train_split, y_train_split = reg(x_train=x_train_norm,
                                                                                  y_train=y_train_norm,
                                                                                  x_test=x_test_norm,
                                                                                  handin=argumenthelper.get_args().handin)
        return regressor, x_test_split, y_test_split, x_train_split, y_train_split

    def predict(self, regressor, x_test_split, y_test_split, x_test_index, x_train_split, y_train_split):
        predicted_values = regressor.predict(x_train_split)
        score = r2_score(y_true=y_train_split, y_pred=predicted_values)
        Logcreator.info("R2 Score achieved on training set: {}".format(score))

        if y_test_split is not None:
            predicted_values = regressor.predict(x_test_split)
            score = r2_score(y_true=y_test_split, y_pred=predicted_values)
            Logcreator.info("R2 Score achieved on test set: {}".format(score))
        else:
            predicted_values = regressor.predict(x_test_split)
            output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(predicted_values.flatten())], axis=1)
            output_csv.columns = ["id", "y"]
            pd.DataFrame.to_csv(output_csv, Configuration.output_directory + '\\submit.csv', index=False)
