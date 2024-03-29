"""
Runs trainings and predictions
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import itertools
import os

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
import matplotlib.pyplot as plt

class Engine:
    def __init__(self):
        Logcreator.info("Training initialized")

    def search(self, x_train, y_train, x_test):

        imputer_par_list = self.get_serach_list('search.imputer')
        outlier_par_list = self.get_serach_list('search.outlier')
        feature_selector_par_list = self.get_serach_list('search.feature_selector')
        normalizer_par_list = self.get_serach_list('search.normalizer')
        regression_par_list = self.get_serach_list('search.regression')

        number_of_loops = len(imputer_par_list) * len(outlier_par_list) * len(feature_selector_par_list) \
                          * len(normalizer_par_list) * len(regression_par_list)

        Logcreator.h1("Number of loops:", number_of_loops)
        loop_counter = 0

        # prepare out columns names
        columns_out = ["Loop_counter", "R2 Score Test", "R2 Score Training"]
        columns_out.extend(['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score'])
        # combine all keys
        columns_out.extend(['imputer_' + s for s in list(imputer_par_list[0].keys())])
        columns_out.extend(['outlier_' + s for s in list(outlier_par_list[0].keys())])
        columns_out.extend(['feature_selector_' + s for s in list(feature_selector_par_list[0].keys())])
        columns_out.extend(['normalizer_' + s for s in list(normalizer_par_list[0].keys())])
        columns_out.extend(['regression_' + s for s in list(regression_par_list[0].keys())])

        # create output dataframe
        results_out = pd.DataFrame(columns=columns_out)
        pd.DataFrame.to_csv(results_out, os.path.join(Configuration.output_directory, 'search_results.csv'), index=False)

        try:
            # TODO clean up for loops
            for imp_data in imputer_par_list:
                # TODO include feature_selector.remove_features_with_many_Nan

                imputer = Imputer(**imp_data)
                x_train_imp, y_train_imp, x_test_imp = imputer.transform_custom(x_train=x_train,
                                                                                y_train=y_train,
                                                                                x_test=x_test)

                for out_data in outlier_par_list:

                    outlier = Outliers(**out_data)
                    x_train_out, y_train_out, x_test_out = outlier.transform_custom(x_train=x_train_imp,
                                                                                    y_train=y_train_imp,
                                                                                    x_test=x_test_imp)

                    for feature_selector_data in feature_selector_par_list:

                        feature_selector = FeatureSelector(**feature_selector_data)

                        x_train_fs, y_train_fs, x_test_fs = feature_selector.transform_custom(x_train=x_train_out,
                                                                                              y_train=y_train_out,
                                                                                              x_test=x_test_out)

                        for normalizer_data in normalizer_par_list:

                            normalizer = Normalizer(**normalizer_data)

                            x_train_norm, y_train_norm, x_test_norm = normalizer.transform_custom(x_train=x_train_fs,
                                                                                                  y_train=y_train_fs,
                                                                                                  x_test=x_test_fs)

                            for regression_data in regression_par_list:
                                # TODO clean up output of current parameters
                                Logcreator.info("\n--------------------------------------")
                                Logcreator.info("Iteration", loop_counter)
                                Logcreator.info("imputer", imp_data)
                                Logcreator.info("outlier", out_data)
                                Logcreator.info("feature_selector", feature_selector_data)
                                Logcreator.info("normalizer", normalizer_data)
                                Logcreator.info("regression", regression_data)
                                Logcreator.info("\n----------------------------------------")

                                regressor = Regression(**regression_data)
                                best_model, x_test_split, y_test_split, x_train_split, y_train_split, search_results = \
                                    regressor.fit_predict(
                                        x_train=x_train_norm, y_train=y_train_norm,
                                        x_test=x_test_norm, handin=False)

                                predicted_values = best_model.predict(x_train_split)
                                score_train = r2_score(y_true=y_train_split, y_pred=predicted_values)
                                Logcreator.info("R2 Score achieved on training set: {}".format(score_train))

                                predicted_values = best_model.predict(x_test_split)
                                score_test = r2_score(y_true=y_test_split, y_pred=predicted_values)
                                Logcreator.info("R2 Score achieved on test set: {}".format(score_test))

                                output = pd.DataFrame()
                                for i in range(0,
                                               5):  # append multiple rows of the grid search result, not just the best
                                    # update output
                                    # TODO not so nice because we only take the values, so the order has to be correct;
                                    #  Maybe converte everything to one dictionary and then append the dictionary to the pandas dataframe;
                                    #  But works for now as long as the order is correct
                                    output_row = list()
                                    output_row.append(loop_counter)
                                    output_row.append(score_test)
                                    output_row.append(score_train)
                                    output_row.extend(search_results[
                                                          ['params', 'mean_test_score', 'std_test_score',
                                                           'mean_train_score',
                                                           'std_train_score']].iloc[i])

                                    output_row.extend(list(imp_data.values()))
                                    output_row.extend(list(out_data.values()))
                                    output_row.extend(list(feature_selector_data.values()))
                                    output_row.extend(list(normalizer_data.values()))
                                    output_row.extend(list(regression_data.values()))
                                    output = output.append(pd.DataFrame(output_row, index=results_out.columns).T)

                                #Write to csv
                                pd.DataFrame.to_csv(output, os.path.join(Configuration.output_directory, 'search_results.csv'), index=False, mode='a', header=False)
                                #Increase loop counter
                                loop_counter = loop_counter + 1
        finally:
            Logcreator.info("Search finished")

    def get_serach_list(self, config_name):
        param_dict = self.get_serach_params(config_name)
        keys, values = zip(*param_dict.items())

        search_list = list()
        # probably there is an easier way to do this
        for instance in itertools.product(*values):
            d = dict(zip(keys, instance))
            search_list.append(d)

        return search_list

    def get_serach_params(self, config_name):
        """
        Not so nice to parse the data from the file to a dictionary...
        Parameters have to have the exact name of the actual class parameter!
        Parameters have to be alphabetically ordered in the config file!
        Parameters can't be named count and index!
        """

        config = Configuration.get(config_name)
        attribute_names = [a for a in dir(config) if not a.startswith('_')]
        # count and index is somehow also an attribute of the config object
        if 'count' in attribute_names:
            attribute_names.remove('count')
        if 'index' in attribute_names:
            attribute_names.remove('index')

        param_dict = dict()
        for name, element in zip(attribute_names, config):
            param_dict[name] = element

        return param_dict

    def train(self, x_train, y_train, x_test):
        # Imputer
        imputer = Imputer(name=Configuration.get('imputer.name'),
                          iterative_n_nearest_features=Configuration.get('imputer.iterative_n_nearest_features'),
                          knn_weights=Configuration.get('imputer.knn_weights'),
                          knn_n_neighbors=Configuration.get('imputer.knn_n_neighbors'))

        x_train_imp, y_train_imp, x_test_imp = imputer.transform_custom(x_train=x_train, y_train=y_train, x_test=x_test)

        # Outliers
        outliers = Outliers(strategy=Configuration.get('outliers.customOR.method'),
                            threshold=Configuration.get('outliers.customOR.threshold'),
                            fit_on=Configuration.get('outliers.fit_on'),
                            name=Configuration.get('outliers.name'),
                            contamination=Configuration.get('outliers.contamination'))
        x_train_outl, y_train_outl, x_test_outl = outliers.transform_custom(x_train=x_train_imp, y_train=y_train_imp, x_test=x_test_imp)

        # Feature selection
        feature_selector = FeatureSelector(
            remove_constant=Configuration.get('feature_selector.remove_constant_features'),
            remove_constant_threshold=Configuration.get('feature_selector.remove_constant_features_par.threshold'),
            remove_correlated=Configuration.get('feature_selector.remove_correlated_features'),
            remove_correlated_threshold=Configuration.get('feature_selector.remove_correlated_features_par.threshold'),
            use_select_best_k=Configuration.get('feature_selector.selectBestK'),
            k=Configuration.get('feature_selector.selectBestK_par.k'),
            use_select_best_based_on_impurity=Configuration.get('feature_selector.selectBestBasedOnImpurity'),
            lda_on=Configuration.get('feature_selector.lda_on'),
            lda_n_components=Configuration.get('feature_selector.lda_n_components'),
            lasso_on=Configuration.get('feature_selector.lasso_on'),
            lasso_alpha=Configuration.get('feature_selector.lasso_alpha'))
        x_train_feat, y_train_feat, x_test_feat = feature_selector.transform_custom(x_train=x_train_outl, y_train=y_train_outl, x_test=x_test_outl)

        # Normalizer
        normalizer = Normalizer(name=Configuration.get('normalizer.name'),
                                fit_on=Configuration.get('normalizer.fit_on'))
        x_train_norm, y_train_norm, x_test_norm = normalizer.transform_custom(x_train=x_train_feat, y_train=y_train_feat, x_test=x_test_feat)

        # Regression
        regression = Regression(name=Configuration.get('regression.name'))
        regressor, x_test_split, y_test_split, x_train_split, y_train_split, search_results = regression.fit_predict(x_train=x_train_norm, y_train=y_train_norm, x_test=x_test_norm, handin=argumenthelper.get_args().handin)

        return regressor, x_test_split, y_test_split, x_train_split, y_train_split, search_results


    def predict(self, regressor, x_test_split, y_test_split, x_test_index, x_train_split, y_train_split):
        predicted_values = regressor.predict(x_train_split)
        score = r2_score(y_true=y_train_split, y_pred=predicted_values)

        self.plot_true_vs_predicted(y_train_split, predicted_values,
                                    title="y-train vs. y-train-predicted",
                                    file="train.png")

        Logcreator.info("R2 Score achieved on training set: {}".format(score))

        if y_test_split is not None:
            predicted_values = regressor.predict(x_test_split)
            score = r2_score(y_true=y_test_split, y_pred=predicted_values)

            self.plot_true_vs_predicted(y_test_split.values.flatten(), predicted_values,
                                        title="y-test vs. y-test-predicted",
                                        file="test.png")

            Logcreator.info("R2 Score achieved on test set: {}".format(score))

        else:
            predicted_values = regressor.predict(x_test_split)
            output_csv = pd.concat([pd.Series(x_test_index.values), pd.Series(predicted_values.flatten())], axis=1)
            output_csv.columns = ["id", "y"]
            pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, 'submit.csv'), index=False)

    def plot_true_vs_predicted(self, y_true, y_predicted, title, file):
        """
        Plotting true vs predicted y values, ordered by the true values.
        """
        y_values_vs_predicted = pd.DataFrame([y_true, y_predicted]).T
        y_values_vs_predicted.sort_values(by=0, inplace=True, axis=0)

        x_index = range(0, y_values_vs_predicted.shape[0])
        fig = plt.figure(figsize=(16, 9), dpi=300)
        fig.suptitle(title)

        plt.scatter(x_index, y_values_vs_predicted[0], s=2)
        plt.scatter(x_index, y_values_vs_predicted[1], s=2)
        fig.savefig(os.path.join(Configuration.output_directory, file))
        plt.show()
