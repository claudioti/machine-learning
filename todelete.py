from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

ML_ALGORITHMS = {"SVM": {'model': svm.SVC(kernel='linear', cache_size=500)},
                 "LR": {'model': LogisticRegression(max_iter=500)},
                 "LDA": {'model': LinearDiscriminantAnalysis()},
                 "KNN": {'model': KNeighborsClassifier()},
                 "CART": {'model': DecisionTreeClassifier()},
                 "NB": {'model': GaussianNB()}}
