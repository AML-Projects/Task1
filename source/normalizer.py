from logcreator.logcreator import Logcreator
from sklearn import preprocessing

class Normalizer:
    def __init__(self):
        Logcreator.info("Start normalizer")


    def standard_scaler(self, x_train, y_train, x_test):
        scaler = preprocessing.StandardScaler()
        x_train_norm = scaler.fit_transform(x_train)
        x_test_norm = scaler.transform(x_test)
        return x_train_norm, y_train, x_test_norm