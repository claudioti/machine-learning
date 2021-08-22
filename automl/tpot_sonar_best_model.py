import time
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from utils import Constants

tpot_data = pd.read_csv(Constants.dataset_path_final, sep=',', dtype=np.float64)
label = tpot_data['Class']
features = tpot_data.drop('Class', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Class'], random_state=1)

# Average CV score on the training set was: 0.9808444444444445
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=12, min_samples_split=7)),
    DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=16, min_samples_split=5)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
start = time.time()
results = exported_pipeline.predict(testing_features)
end = time.time()

print("Accuracy: %f" % accuracy_score(testing_target, results))
print("Precision: %f" % precision_score(testing_target, results, average="macro"))
print("Recall: %f" % recall_score(testing_target, results, average="macro"))
print("F1 Score: %f" % f1_score(testing_target, results, average="macro"))
print("Time (sec): %f" % (end - start))

#######Cross Validation

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': 'precision',
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score)}

# feature = tpot_data[[
#     'NumericRatio', 'ASN', 'DomainLength', 'NumericSequence', 'Ip', 'ConsoantRatio', 'TLD', 'StrangeCharacters', 'CountryCode']]

kf = model_selection.KFold(n_splits=10, random_state=None)
result = cross_validate(exported_pipeline, features, label, cv=kf, scoring=scoring, return_estimator=True)
print()
print("Cross Validation")
print("Accuracy: %f" % result['test_accuracy'].mean())
print("Precision: %f" % result['test_precision'].mean())
print("Recall: %f" % result['test_recall'].mean())
print("F1 Score: %f" % result['test_f1'].mean())
print("Time (sec): %f" % result['score_time'].mean())

meanFeatureValues = {}

for idx, estimator in enumerate(result['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.steps[0][1].estimator.feature_importances_,
                                       index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    for index, row in feature_importances.iterrows():
        if index not in meanFeatureValues:
            meanFeatureValues[index] = row.importance
        else:
            meanFeatureValues[index] = meanFeatureValues[index] + row.importance
    print(feature_importances)

print()
meanFeatureValues = dict(sorted(meanFeatureValues.items(), key=lambda x: x[1], reverse=True))
for feature in meanFeatureValues:
    print("%s: %s" % (feature, str('{:.10f}'.format(meanFeatureValues[feature] / 10))))