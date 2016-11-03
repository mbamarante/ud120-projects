#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
### accuracies here consider sliced training set
### A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.
#clf = SVC(kernel="linear") # accuracy: 0.88
#clf = SVC(kernel="rbf") # accuracy: 0.616040955631
#clf = SVC(kernel="rbf", C=10.0) # accuracy: 0.616040955631
#clf = SVC(kernel="rbf", C=100.0) # accuracy: 0.616040955631
#clf = SVC(kernel="rbf", C=1000.0) # accuracy: 0.821387940842
clf = SVC(kernel="rbf", C=10000.0) # accuracy: 0.892491467577

### smaller training set to reduce time...
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#### store your predictions in a list named pred
t0 = time()
pred = clf.predict(features_test)
print "training time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print acc

# print clf.predict(features_test[10]), clf.predict(features_test[26]), clf.predict(features_test[50])

import numpy
print numpy.sum(pred)
#########################################################


