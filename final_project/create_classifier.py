#!/usr/bin/python

import sys
import numpy as np
sys.path.append("../tools/")
from sklearn.grid_search import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit

# Performance > .3
# GaussianNB               Precision: 0.50902	Recall: 0.39500
# DicisionTree             Precision: 0.30242	Recall: 0.31300
# AdaBoost                 Precision: 0.30717	Recall: 0.31700
# GaussianNB/PCA           Precision: 0.55469	Recall: 0.39050

# GaussianNB/PCA           4              10      0.24971	0.10600
# GaussianNB/PCA           6              10      0.21106	0.11450
# SVC                      4              10            -         -
# SVC/Tunned               4              10            -         -
# DicisionTree             4              10      0.29685    0.33500
# DicisionTree             4              0       0.26392	 0.31050
# DicisionTree             6              10      0.26555	 0.28600
# DecisionTree/Tunned      4              10      0.38840	 0.17750
# KNeighbors               4              10      0.56881	 0.37200    <-
# KNeighbors/Tunned        4              10      0.64105	 0.34200    <-
# KNeighbors/Tunned        6              10      0.41705	 0.18100
# KNeighbors/Tunned        3              10      0.55895	 0.20150
# KNeighbors/Tunned        4               0      0.36731	 0.10450
# AdaBoost                 4              10      0.29393	 0.32950
# RandomForest             4              10      0.31516	 0.11850

def new_classifier():
    return new_gaussian_nb()
    # return new_svc()
    # return new_decision_tree()
    # return new_k_neighbors()
    # return new_ada_boost_classifier()
    # return new_random_forest()

def new_tunned_classifier(my_dataset, features_list):
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    return new_tunned_gaussian_nb()
    # return new_tunned_svc(features, labels)
    # return new_tunned_decision_tree(features, labels)
    # return new_tunned_k_neighbors(features, labels)

def new_gaussian_nb():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()

def new_tunned_gaussian_nb():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    return Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA(n_components=3)),
        ('classification', GaussianNB())
    ])

def new_svc():
    from sklearn.svm import SVC
    return SVC()

def new_tunned_svc(features, labels):
    from sklearn.svm import SVC
    parameters = {'kernel': ['rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    clf_grid = GridSearchCV(SVC(), parameters)
    clf_grid.fit(features, labels)
    return clf_grid.best_estimator_

def new_decision_tree():
    from sklearn import tree
    return tree.DecisionTreeClassifier()

def new_tunned_decision_tree(features, labels):
    from sklearn import tree
    parameters = {
        'min_samples_split': [5, 10, 20, 30, 40],
        'max_features': ["auto", "sqrt", "log2"],
        'min_samples_leaf': [2, 4, 6, 8],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 2, 5, 10],
        "max_leaf_nodes": [None, 5, 10, 20]
    }
    clf_grid = GridSearchCV(tree.DecisionTreeClassifier(), parameters)
    clf_grid.fit(features, labels)
    return clf_grid.best_estimator_

def new_k_neighbors():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=3, leaf_size=2, weights="distance")

def new_tunned_k_neighbors(features, labels):
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    from sklearn.neighbors import KNeighborsClassifier
    metrics = ['minkowski', 'euclidean', 'manhattan']
    weights = ['distance']
    leaf_size = [2, 3, 4, 5, 7, 10, 15]
    numNeighbors = np.arange(2, 4)
    param_grid = dict(metric=metrics, leaf_size=leaf_size, weights=weights, n_neighbors=numNeighbors)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
    grid.fit(features, labels)
    return grid.best_estimator_

def new_ada_boost_classifier():
    from sklearn import tree
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(tree.DecisionTreeClassifier(),
                             algorithm="SAMME",
                             n_estimators=50)

def new_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier()