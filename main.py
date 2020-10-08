__author__ = 'Andreas Kaufmann'
__email__ = "ankaufmann@student.ethz.ch"

import os
import time
import argparse
import pandas as pd
from shutil import copy

from source.configuration import Configuration
from source.engine import Engine
from logcreator.logcreator import Logcreator
from helpers import argumenthelper


if __name__ == "__main__":
    global config
    #Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--handin', default=False, type=bool, help="If set to true, whole trainingset used for training")

    #args = parser.parse_args()
    args = argumenthelper.parse_args(parser)
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir)
    Logcreator.initialize()

    Logcreator.h1("Task 01 - Regression")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    # Load training data
    x_train = pd.read_csv("./data/X_train.csv")
    x_train.drop(['id'], axis=1, inplace=True)
    Logcreator.info("Shape of training_samples: {}".format(x_train.shape))
    Logcreator.info(x_train.head())

    y_train = pd.read_csv("./data/y_train.csv")
    y_train.drop(['id'], axis=1, inplace=True)
    Logcreator.info("Shape of training labels: {}".format(y_train.shape))
    Logcreator.info(y_train.head())

    x_test = pd.read_csv("./data/X_test.csv")
    x_test.drop(['id'], axis=1, inplace=True)
    Logcreator.info("Shape of test samples: {}".format(x_test.shape))
    Logcreator.info(y_train.head())

    engine = Engine()

    #Train
    regressor, x_test_split, y_test_split = engine.train(x_train=x_train, y_train=y_train, x_test=x_test)
    #Predict
    if not args.handin:
        engine.predict(regressor=regressor, x_test_split=x_test_split, y_test_split=y_test_split, x_test_index=None)
    else:
        engine.predict(regressor=regressor, x_test_split=x_test_split, y_test_split=None, x_test_index=x_test.index)

    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
