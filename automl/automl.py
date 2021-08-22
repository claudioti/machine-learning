

# define dataset
from pandas import np
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier

from utils import functions, Constants
from torch import nn
import pandas as pd

dataset = np.loadtxt(Constants.dataset_path_final,delimiter=',',skiprows=1)
#dataset = pd.read_csv(Constants.dataset_path_final)
#dataset = functions.read_data(path=Constants.dataset_path_final, fillna=True, normalization=False)
##DROPING IP COLUMN
#dataset = np.delete(dataset, 5, 1)
X = dataset[:, :-1]
y = dataset[:, -1]
y = y.astype(int)

# define model evaluation
##cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search
##model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=1)
model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=4, max_eval_time_mins=2)

# perform the search
model.fit(X, y)

# export the best model
model.export('C:\\Mestrado\\machine-learning\\tpot_sonar_best_model_new.py')



