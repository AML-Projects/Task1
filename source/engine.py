from logcreator.logcreator import Logcreator
from source.configuration import Configuration
import pandas as pd
from sklearn.metrics import r2_score
from source.imputers import Imputer
from source.outliers import Outliers
from source.normalizer import Normalizer
from source.regression import Regression
import os

class Engine:
    def __init__(self):
        Logcreator.info("Training initialized")

    def train(self, x_train, y_train, x_test):

        #Imputer
        imputer = Imputer()
        imputer_type = Configuration.get('imputer.name')
        switcher = {
            'mean': imputer.mean_simple_imputer,
            'iterative': imputer.multivariate_imputer
        }
        imp = switcher.get(imputer_type)
        x_train_imp, y_train_imp, x_test_imp = imp(x_train=x_train, y_train=y_train, x_test=x_test)

        #Outliers
        outliers = Outliers()
        outliers_method = Configuration.get('outliers.name')
        switcher = {
            'lof': outliers.LOF
        }
        outl = switcher.get(outliers_method)
        x_train_outl, y_train_outl, x_test_outl = outl(x_train=x_train_imp, y_train=y_train_imp, x_test=x_test_imp)

        #Normalizer
        normalizer = Normalizer()
        normalizer_method = Configuration.get('normalizer.name')
        switcher = {
            'stdscaler': normalizer.standard_scaler
        }
        norm = switcher.get(normalizer_method)
        x_train_norm, y_train_norm, x_test_norm = norm(x_train=x_train_outl, y_train=y_train_outl, x_test=x_test_outl)

        #Regression
        regression = Regression()
        regression_method = Configuration.get('regression.name')
        switcher = {
            'ridge': regression.ridge_regression
        }
        reg = switcher.get(regression_method)
        regressor, x_test_split, y_test_split = reg(x_train=x_train_norm, y_train=y_train_norm, x_test=x_test_norm, split=Configuration.get('regression.split'))
        return regressor, x_test_split, y_test_split

    def predict(self, regressor, x_test_split, y_test_split, x_test_index):
        if y_test_split is not None:
            predicted_values = regressor.predict(x_test_split)
            score = r2_score(y_true=y_test_split, y_pred=predicted_values)
            Logcreator.info("Score achieved on test-set: {}".format(score))
        else:
            predicted_values = regressor.predict(x_test_split)
            output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(predicted_values.flatten())], axis=1)
            output_csv.columns = ["id", "y"]

            pd.DataFrame.to_csv(output_csv, Configuration.output_directory, index=False)