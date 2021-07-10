# Copyright (C) 2020 Claudio Marques - All Rights Reserved
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

dataset_path = "data/output/dataset{toReplace}.csv"
dataset_path_final = "C:\\Mestrado\\machine-learning\\data\\input\\BenignAndMaliciousDataset_LabelEncoded_Normalized.csv"


fileHeader = "Domain,MXDnsResponse,TXTDnsResponse,HasSPFInfo,HasDkimInfo,HasDmarcInfo,Ip,DomainInAlexaDB,CommonPorts,CountryCode,RegisteredCountry,CreationDate," \
             "LastUpdateDate,ASN,HttpResponseCode,RegisteredOrg,SubdomainNumber,Entropy,EntropyOfSubDomains,StrangeCharacters," \
             "TLD,IpReputation,DomainReputation," \
             "ConsoantRatio,NumericRatio,SpecialCharRatio,VowelRatio,ConsoantSequence,VowelSequence,NumericSequence,SpecialCharSequence,DomainLength,Class"

headerRegex = "%s,%d,%d,%d,%d,%d,%s,%d,%d,%s,%s,%d," \
              "%d,%d,%d,%s,%d,%d,%d,%d," \
              "%s,%d,%d," \
              "%0.1f,%0.1f,%0.1f,%0.1f,%d,%d,%d,%d,%d,%d\n"

# sublist3rEngines = "baidu,yahoo,bing,ask,netcraft,dnsdumpster,threatcrowd,ssl,passivedns"
sublist3rEngines = "bing,passivedns"

###EXPLORATORY
describe_png_path = "data/output/exploratory/describe/describe-{toReplace}.png"
info_png_path = "data/output/exploratory/info/info-{toReplace}.png"
info_groupby_png_path = "data/output/exploratory/info/groupby/info-{toReplace}.png"

###ML
#predictionFields = ['NumericRatio', 'NumericSequence', 'DomainLength', 'HasSPFInfo', 'TXTDnsResponse', 'ConsoantRatio']
predictionFields = ['SpecialCharRatio', 'VowelSequence', 'SubdomainNumber', 'SpecialCharSequence', 'HttpResponseCode', 'VowelRatio', 'ConsoantRatio']
# MLAlgorithms = [{
#     "name": "Logistic Regression",
#     "type": "LR"
# },
#     {
#         "name": "Decision Tree",
#         "type": "DT"
#     },
#     {
#         "name": "Random Forest",
#         "type": "RF"
#     },
#     {
#         "name": "NaiveBayes",
#         "type": "NB"
#     }
# ]
outcomeVar = "Class"
# {
#     "name": "Support Vector Machines",
#     "type": "SVM",
#     "predictionFields": predictionFields
# }

MLAlgorithms = [{
    "name": "Support Vector Machines",
    "type": "SVM"
}]


#####
TRAIN_PATH = "C:\\Mestrado\\machine-learning\\data\\train-data\\"
MODEL_OUTPUT_PATH = "C:\\Mestrado\\machine-learning\\data\\model\\"
TABLES_OUTPUT_PATH = "C:\\Mestrado\\machine-learning\\data\\output\\"
ML_ALGORITHMS = {"SVM": {'enable': True, 'model': svm.SVC(kernel='linear', cache_size=500), 'trained_model': None},
                 "LR": {'enable': True, 'model': LogisticRegression(max_iter=500), 'trained_model': None},
                 "LDA": {'enable': True, 'model': LinearDiscriminantAnalysis(), 'trained_model': None},
                 "KNN": {'enable': True, 'model': KNeighborsClassifier(), 'trained_model': None},
                 "CART": {'enable': True, 'model': DecisionTreeClassifier(), 'trained_model': None},
                 "NB": {'enable': True, 'model': GaussianNB(), 'trained_model': None},
                 "AutoML-CART": {'enable': False, 'model': DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=6, min_samples_split=13), 'trained_model': None}}