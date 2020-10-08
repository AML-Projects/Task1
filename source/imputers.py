from logcreator.logcreator import Logcreator
from source.configuration import Configuration
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class Imputer:
    def __init__(self):
        Logcreator.info("Imputer initialized")


    def mean_simple_imputer(self, x_train, y_train, x_test):
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(x_train)
        x_train_imputed = imp.transform(x_train)
        x_test_imputed = imp.transform(x_test)
        Logcreator.info("Imputation reslut with mean imputation for x_train: \n")
        Logcreator.info(pd.DataFrame(x_train_imputed).head())
        Logcreator.info("Imputation reslut with mean imputation for x_test: \n")
        Logcreator.info(pd.DataFrame(x_test_imputed).head())
        return x_train_imputed, y_train, x_test_imputed

    def multivariate_imputer(self, x_train, y_train, x_test):
        Logcreator.info("Nr. of rows without any NaN: {} \n".format(x_train.shape[0] - sum([True for idx, row in x_train.iterrows() if any(row.isna())])))

        imp = IterativeImputer(missing_values=np.nan, max_iter=1, sample_posterior=False, random_state=0)
        imp.fit(x_train)
        Logcreator.info(x_train.head())

        IterativeImputer(random_state=0)
        x_train_imputed = imp.transform(x_train)
        x_test_imputed = imp.transform(x_test)
        return x_train_imputed, y_train, x_test_imputed