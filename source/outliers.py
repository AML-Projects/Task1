from logcreator.logcreator import Logcreator
from sklearn.neighbors import LocalOutlierFactor

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