from logcreator.logcreator import Logcreator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge
from sklearn import model_selection
import math

class Regression:
    def __init__(self):
        Logcreator.info("Start fit model")


    def ridge_regression(self, x_train, y_train, x_test, split):
        if split:
            x_train_split, x_test_split, y_train_split, y_test_split = model_selection.train_test_split(
            x_train, y_train, test_size=0.1)
        else:
            x_train_split = x_train
            x_test_split = x_test
            y_train_split = y_train
            y_test_split = None
        Logcreator.info("x_train_split: {}".format(x_train_split.shape))
        Logcreator.info("x_test_split: {}".format(x_test_split.shape))

        # Runn Gridsearch for alpha
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
        return ridge, x_test_split, y_test_split