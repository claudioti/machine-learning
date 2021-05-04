from pandas import np

from utils import functions, Constants
from ast import literal_eval
import glob, os


def menu_create_train_and_test_data(dataset=None):
    if dataset is None:
        raise Exception("Please read a dataset before continue...")
    selected_features = input(
        "Please insert an array of features or leave empty to select all (ie: [\"FirstFeature\", \"SecondFeature\"]: \n")
    if len(selected_features) == 0:
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
    for file in os.listdir(Constants.MODEL_OUTPUT_PATH ):
        if file.endswith(".save"):
            print(file)
    suffix = input("\nPlease insert the suffix from the list above to be load (ie: all_features_test_size_0.2): \n")
    return functions.load_models(ml_algorithms, suffix)
