# import the necessary packages
from __future__ import print_function
from sklearn import svm
from sklearn.metrics import classification_report
from pandas import read_csv
import numpy as np
import sklearn
 
# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
	from sklearn.cross_validation import train_test_split
 
# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split
 
# load data
# Data Source: https://archive.ics.uci.edu/ml/datasets/Turkiye+Student+Evaluation
data = read_csv("turkiye-student-evaluation_R_Specific.csv")
data.target = data.values[:,1]
 
# training and testing split, using 75% of the
# data for training and 25% for testing

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data),
	data.target, test_size=0.25, random_state=42)
 

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("testing data points: {}".format(len(testLabels)))

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []
 
model = svm.SVC() #(n_neighbors=k)
model.fit(trainData, trainLabels)

# Score of Training Data: R Squared
score = model.score(trainData, trainLabels)


model.fit(trainData, trainLabels)
predictions = model.predict(testData)
 
# Score of Testing Data: Accuancy of SVM Model
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))
