import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import submenus
from utils import Constants, functions
import pandas as pd

# TEST AND TRAIN CONFIGS
TEST_SIZE = 0.2
SELECTED_FEATURES = None

# ML/Model CONFIGS
ML_ALGORITHMS = Constants.ML_ALGORITHMS
# models.append(('SVM', svm.SVC()))
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))

# Menu variables
dataset = None
features_train, features_test, class_train, class_test, load_data_filename = None, None, None, None, None


def menu():
    # Global variable
    global ML_ALGORITHMS
    global dataset
    global SELECTED_FEATURES
    global features_train, features_test, class_train, class_test, load_data_filename

    print("************ML Train and Test**************")
    print()

    choice = input("""
                      1: Read Dataset
                      2: Split Dataset (test and train)
                      3: Load test and train data
                      4: Train Models
                      5: Save Models
                      6: Load Models
                      7: Test Models
                      8: Feature Selection
                      0: Exit

                      Please enter your choice: """)
    choice = int(choice)
    if choice == 0:
        exit(0)
    elif choice == 1:
        # Read dataset
        dataset = functions.read_data(path=None, fillna=False, normalization=False)
    elif choice == 2:
        # Create test and train data
        submenus.menu_create_train_and_test_data(dataset, SELECTED_FEATURES)
    elif choice == 3:
        # Load test and train data
        features_train, features_test, class_train, class_test, load_data_filename = submenus.menu_load_train_and_test_data()
    elif choice == 4:
        # Train
        ML_ALGORITHMS = functions.train_model(ml_algorithms=ML_ALGORITHMS, features_train=features_train,
                                              class_train=class_train)
    elif choice == 5:
        # if load_data_filename is None:
        #     print("Please load test and train data first.")
        #     menu()
        # Save models
        functions.save_models(ML_ALGORITHMS, Constants.MODEL_OUTPUT_PATH, load_data_filename)
    elif choice == 6:
        # Load models
        ML_ALGORITHMS = submenus.menu_load_models(ML_ALGORITHMS)
    elif choice == 7:
        # Test models
        ML_ALGORITHMS = submenus.menu_test_models(ML_ALGORITHMS, features_test=features_test, class_test=class_test, filename=load_data_filename, dataset=dataset, selected_features=SELECTED_FEATURES)
    elif choice == 8:
        # Feature Importance
        if dataset is None:
            print("Please read the dataset before continue.")
            menu()
        SELECTED_FEATURES = submenus.menu_feature_selection(dataset)
    else:
        print("You must only select a number from the menu")
        print("Please try again")
        menu()
    menu()
