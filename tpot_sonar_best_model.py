from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import matplotlib.pyplot as plt

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from utils import Constants

tpot_data = pd.read_csv(Constants.dataset_path_final, sep=',', dtype=np.float64)
features = tpot_data.drop('Class', axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['Class'], random_state=1)

# Average CV score on the training set was: 0.9799629629629629
exported_pipeline = make_pipeline(
    StackingEstimator(
        estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling",
                                loss="modified_huber", penalty="elasticnet", power_t=0.5)),
    DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=6, min_samples_split=13)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)
#
# Start Evaluation
#
t0 = time()
exported_pipeline.fit(training_features, training_target)
print('training time: ', round(time() - t0, 3), 's')

t1 = time()
results = exported_pipeline.predict(testing_features)
print('predicting time: ', round(time() - t1, 3), 's')
accuracy = accuracy_score(testing_target, results)

print('Confusion Matrix: ')

print(confusion_matrix(testing_target, results))

# Accuracy in the 0.9333, 9.6667, 1.0 range

print("Accuracy: " + str(accuracy))

coef = exported_pipeline.named_steps['stackingestimator'].estimator.coef_
feature_selection = []
for i in range(0, coef.shape[0]):
    top_indices = np.argsort(coef[i])[-9:]
for i in top_indices:
    feature_selection.append(training_features.columns[i])

print("Used features: " + str(feature_selection))

plot_confusion_matrix(exported_pipeline, testing_features, testing_target)
plt.show()
