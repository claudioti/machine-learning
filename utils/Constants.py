# Copyright (C) 2020 Claudio Marques - All Rights Reserved
dataset_path = "data/output/dataset{toReplace}.csv"
dataset_path_final = "C:\\SmithMicro\\machineLearning\\data\\input\\BenignAndMaliciousDataset_LabelEncoded.csv"
#dataset_path_final = "data/output/final/datasetFinal.csv"
# log_path = "data/logs/output.log"
log_path = "data/logs/output_append.log"
numberOfThreads = 45

inputFileMalign = "data/input/malign/all.log"
outputFileMalign = "data/output/fileMalign.csv"
# sampleMalign = 500 #8500
sampleMalign = 0

inputFileBenignAAAA = "C:\\Mestrado\\python-trab-final\\data\\input\\benign\\aaaa\\all.log"
outputFileBenignAAA = "data/output/fileBenignAAAA.csv"
sampleAAAA = 45
# sampleAAAA = 1

inputFileBenignCNAME = "C:\\Mestrado\\python-trab-final\\data\\input\\benign\\cname\\all.log.2"
outputFileBenignCNAME = "data/output/fileBenignCNAME.csv"
sampleCNAME = 0
# sampleCNAME = 1

inputFileBenignMX = "C:\\Mestrado\\python-trab-final\\data\\input\\benign\\mx\\all.log"
outputFileBenignMX = "data/output/fileBenignMX.csv"
sampleMX = 0
# sampleMX = 1

alexaDbPath = "C:\\Mestrado\\python-trab-final\\utils\\Database\\AlexaDB\\top-1m.csv"

ports = [80, 443, 21, 22, 23, 25, 53, 110, 143, 161, 445, 465, 587, 993, 995, 3306, 3389, 7547, 8080, 8888]

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
TRAIN_PATH = "C:\\SmithMicro\\machineLearning\data\\train-data\\"
MODEL_OUTPUT_PATH = "C:\\SmithMicro\\machineLearning\data\\model\\"
