#!/usr/bin/python

import sys
sys.path.append("../tools/")

import matplotlib.pyplot
import numpy as np
import pprint
from feature_format import featureFormat, targetFeatureSplit

def do_validate(clf, K, my_dataset, features_list):
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    evaluate_cross_validation(clf, features, labels, K)

def evaluate_cross_validation(clf, X, y, K):
    from sklearn.cross_validation import KFold
    from sklearn.cross_validation import cross_val_score
    cv = KFold(len(y), K, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), scores.std()))
