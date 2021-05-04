from pandas import np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2

from utils import functions, Constants
from ast import literal_eval
import glob, os
import pandas as pd
import matplotlib.pyplot as plt


def menu_create_train_and_test_data(dataset=None, selected_features=None):
    if dataset is None:
        raise Exception("Please read a dataset before continue...")
    if selected_features is None:
        choice = input(
            "Please insert an array of features or leave empty to select all (ie: [\"FirstFeature\", \"SecondFeature\"]: \n")
        if len(choice) == 0:
            selected_features = None

    test_size = input("Please insert the percentage of test size in a float format (ie: 0.2): \n")
    if type(test_size) is not float:
        try:
            test_size = float(test_size)
        except Exception as ex:
            raise Exception("Test size is not a valid float!\n" + str(ex))

    X_train, X_test, y_train, y_test = functions.create_train_data(data=dataset, selected_features=selected_features,
                                                                   test_size=test_size)
    print("Train and Test data created!\n")
    choice = input("Do you want to save the train and test data (Y|N)? \n")
    if choice == "Y":
        data_filename = input("Please insert the filename to be saved? \n")
        np.save(Constants.TRAIN_PATH + data_filename + ".npy", (X_train, X_test, y_train, y_test))


def menu_load_train_and_test_data():
    print("\nList of available train and test data:\n")
    for file in os.listdir(Constants.TRAIN_PATH):
        if file.endswith(".npy"):
            print(file)
    choice = input("\nPlease insert the filename from the list above to be load (include extension): \n")
    features_train, features_test, class_train, class_test = functions.load_data(path=Constants.TRAIN_PATH + choice)
    # Load test and train data
    return features_train, features_test, class_train, class_test, choice.replace(".npy", "")


def menu_load_models(ml_algorithms):
    print("\nList of available train and test data:\n")
    for file in os.listdir(Constants.MODEL_OUTPUT_PATH):
        if file.endswith(".save"):
            print(file)
    suffix = input("\nPlease insert the suffix from the list above to be load (ie: all_features_test_size_0.2): \n")
    return functions.load_models(ml_algorithms, suffix)


def menu_feature_selection(dataset):
    X = dataset[[
        "IpSplit1", "IpSplit2", "IpSplit3", "IpSplit4", "MXDnsResponse", "TXTDnsResponse", "HasSPFInfo", "HasDkimInfo",
        "HasDmarcInfo", "DomainInAlexaDB", "CommonPorts", "CountryCode", "RegisteredCountry", "CreationDate",
        "LastUpdateDate", "ASN", "HttpResponseCode", "RegisteredOrg", "SubdomainNumber", "Entropy",
        "EntropyOfSubDomains", "StrangeCharacters", "TLD", "IpReputation", "DomainReputation", "ConsoantRatio",
        "NumericRatio", "SpecialCharRatio", "VowelRatio", "ConsoantSequence", "VowelSequence", "NumericSequence",
        "SpecialCharSequence", "DomainLength"]]  # independent columns
    y = dataset["Class"]  # target column

    def univariate_selection():
        # apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Features', 'Score']  # naming the dataframe columns
        print(featureScores.nlargest(10, 'Score'))
        print(str(featureScores.nlargest(10, 'Score').Features.values))
        return featureScores.nlargest(10, 'Score').Features.values

    def feature_importance():
        model = ExtraTreesClassifier()
        model.fit(X, y)
        # print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        # plt.show()
        print(str(feat_importances.nlargest(10).axes[0].values))
        return feat_importances.nlargest(10).axes[0].values

    print()
    choice = input(""" Select the method to the feature selection:
                      1: Univariate Selection
                      2: Feature Importance
                      0: Main Menu

                      Please enter your choice: """)
    choice = int(choice)
    if choice == 0:
        return None
    elif choice == 1:
        return univariate_selection()
    elif choice == 2:
        return feature_importance()
    else:
        print("You must only select a number from the menu")
        print("Please try again")
        menu_feature_selection(dataset)


def menu_test_models(ML_ALGORITHMS, features_test, class_test, filename, dataset, selected_features):
    choice = input(""" Select the method to test the models:
                      1: Single test
                      2: K-cross validation
                      0: Main Menu

                      Please enter your choice: """)
    choice = int(choice)
    if choice == 0:
        return None
    elif choice == 1:
        functions.single_test(ML_ALGORITHMS, features_test, class_test, filename)
        menu_test_models(ML_ALGORITHMS, features_test, class_test, filename, dataset, selected_features)
    elif choice == 2:
        functions.kcross_validation(ML_ALGORITHMS, filename, dataset, selected_features)
        menu_test_models(ML_ALGORITHMS, features_test, class_test, filename, dataset, selected_features)
    else:
        print("You must only select a number from the menu")
        print("Please try again")
        menu_test_models(ML_ALGORITHMS, features_test, class_test, filename, dataset, selected_features)
