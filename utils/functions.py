import os
from datetime import datetime

import pandas as pd
import numpy as np
import pickle
import time

from sklearn import metrics, model_selection
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from tabulate import tabulate

from utils import Constants

# Table variables
TABLE_HEADER = ["Algorithm", "N# Features", "Test Size", "Train Size", "Accuracy", "Precision", "Recall", "Time (sec)"]
TABLE_HEADER_K_CROSS = ["Algorithm", "Test Mode", "N# Features", "Accuracy", "Precision",
                        "Recall", "Time (sec)"]


def read_data(path=None, fillna=True, normalization=True):
    df = None
    if path is None:
        print("read_data: path is None.")
        path = Constants.dataset_path_final
        df = pd.read_csv(path)
    else:
        print("read_data: path is not None.")
        df = pd.read_csv(path)

    if fillna:
        print("read_data: fillna True.")
        df = df.fillna('null')

    if normalization:
        print("read_data: normalization True.")
        df = get0and1FromDatatype(df, "int64", ["DomainLength", "NumericSequence", "IpSplit1", "IpSplit2", "IpSplit3",
                                                "IpSplit4", "CountryCode", "RegisteredCountry", "CreationDate",
                                                "LastUpdateDate",
                                                "ASN", "HttpResponseCode", "RegisteredOrg", "SubdomainNumber",
                                                "Entropy",
                                                "EntropyOfSubDomains",
                                                "StrangeCharacters", "TLD", "ConsoantSequence", "VowelSequence",
                                                "SpecialCharSequence"])
    return df


def get0and1FromDatatype(df, datatype, columns):
    for col in columns:
        if df[col].dtype.name == datatype:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def create_train_data(data, selected_features, test_size):
    outcome_column = data['Class']
    data = data.drop(columns=['Domain', 'Class'])

    if selected_features is not None:
        data = data[selected_features]

    return train_test_split(data, outcome_column,
                            test_size=test_size)  # 80% training and 20% test


def load_data(path):
    validation = input('Do you want to load the train and test data at [' + str(path) + '] (Y|N)?\n')
    if validation == "Y":
        return np.load(path, allow_pickle=True)


def train_model(ml_algorithms, features_train, class_train):
    for key, value in ml_algorithms.items():
        if value['enable']:
            print("\nProcessing the algorithm: " + str(key))
            value['trained_model'] = train_model_impl(model=value['model'], features_train=features_train,
                                                      class_train=class_train)
    return ml_algorithms


def train_model_impl(model, features_train, class_train):
    return model.fit(features_train, class_train)


def save_models(ml_algorithms, path, load_data_filename):
    for key, value in ml_algorithms.items():
        if value['enable']:
            save_model = input('Do you want to save the model ' + str(key) + ' (Y|N)?\n')
            if save_model == "Y":
                filename = path + load_data_filename + "_" + str(key) + ".save"
                pickle.dump(value['trained_model'], open(filename, 'wb'))


def load_models(ml_algorithms, suffix):
    for key, value in ml_algorithms.items():
        if value['enable']:
            filename = Constants.MODEL_OUTPUT_PATH + suffix + "_" + str(key) + ".save"
            value['trained_model'] = pickle.load(open(filename, 'rb'))
            print("\nModel " + suffix + "_" + str(key) + ".save" + " loaded!")
    return ml_algorithms


def single_test(ml_algoritms, features_test, class_test, filename):
    table_data = []

    for key, value in ml_algoritms.items():
        if value['enable']:
            print("\nTesting the " + str(key) + " model.")
            model = value['trained_model']
            # Predict the response for test dataset
            start_time = time.clock()
            class_pred = model.predict(features_test)
            end_time = time.clock() - start_time

            # Model Accuracy: how often is the classifier correct?
            print("Accuracy:", metrics.accuracy_score(class_test, class_pred))

            # Model Precision: what percentage of positive tuples are labeled as such?
            print("Precision:", metrics.precision_score(class_test, class_pred))

            # Model Recall: what percentage of positive tuples are labelled as such?
            print("Recall:", metrics.recall_score(class_test, class_pred))

            # Time to test
            print("Time(sec): " + str(end_time))

            table_data.append([str(key), len(features_test.columns), len(features_test), 90000 - len(features_test),
                               metrics.accuracy_score(class_test, class_pred),
                               metrics.precision_score(class_test, class_pred),
                               metrics.recall_score(class_test, class_pred), str(end_time)])

    print_results_table([TABLE_HEADER], table_data, filename, features_test.columns)


def kcross_validation(ml_algoritms, filename, dataset, selected_features):
    table_data_k_cross = []
    if selected_features is None:
        print("Please use the feature selection to select features before continue...")
        return None

    features = dataset[selected_features]
    label = dataset['Class']

    k = input("Please enter a integer to K value (ie: 10):")
    try:
        k = int(k)
    except Exception as ex:
        print("Invalid K value" + str(ex))
        kcross_validation(ml_algoritms, filename, dataset, selected_features)

    for key, value in ml_algoritms.items():
        if value['enable']:
            print("\nTesting the " + str(key) + " model.")
            model = value['model']
            ##Cross Validation
            # TODO enviar todo o dataset
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': 'precision',
                       'recall': make_scorer(recall_score),
                       'f1': make_scorer(f1_score)}
            kf = model_selection.KFold(n_splits=k, random_state=None)
            start_time = time.clock()
            result = cross_validate(model, features, label, cv=kf, scoring=scoring)
            end_time = time.clock() - start_time
            table_data_k_cross.append([str(key), str(k) + "-fold-cross-validation", len(features.columns),
                                       result['test_accuracy'].mean(),
                                       result['test_precision'].mean(),
                                       result['test_recall'].mean(), str(end_time)])

    if filename is None:
        filename = input("Please insert the filename to save the table data: ")
    print_results_table([TABLE_HEADER_K_CROSS], table_data_k_cross, filename, features.columns)


def print_results_table(table_header, table_data, filename, features):
    table = table_header
    for data in table_data:
        table.append(data)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    choice = input("Do you want to save the table to a file (Y|N)?")
    if choice == "Y":
        write_to_file(filename=filename, content=tabulate(table, headers='firstrow', tablefmt='fancy_grid'),
                      features=features)


def write_to_file(path=None, filename=None, content="", features=None):
    if path is None:
        path = Constants.TABLES_OUTPUT_PATH
    if filename is None:
        raise Exception("No filename found!")
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    separator = "-------------------------------------------------------------------------------------------------------------------"

    with open(path + filename + ".txt", "a", encoding="utf-8") as f:
        f.write('\n'
                + separator
                + '\n'
                + "Features: "
                + str(features)
                + '\n'
                + dt_string
                + ':\n' + content)
        f.close()
    print("Content appended to file at " + path + filename + ".txt")
