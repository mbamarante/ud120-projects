#!/usr/bin/python

import sys
import operator
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

def print_features_score(data_dict, features_list):
    fs = get_featured_score(data_dict, features_list)
    features_scored = dict(zip(features_list[1:], fs.scores_))
    pprint.pprint(sorted(features_scored.items(), key=operator.itemgetter(1)))

def get_features(data_dict, features_list, sel_features_number=5):
    fs = get_featured_score(data_dict, features_list, features_number=sel_features_number)
    features_scored =  dict(zip(features_list[1:], fs.get_support()))
    features_selected = ['poi']
    for feature_scored in features_scored:
        if features_scored[feature_scored] and not feature_scored in features_selected:
            features_selected.append(feature_scored)
    return features_selected

def get_featured_score(data_dict, features_list, features_number=5):
    data = featureFormat(data_dict, features_list, sort_keys = False)
    labels, features = targetFeatureSplit(data)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    fs = SelectKBest(f_classif, features_number)
    fs.fit_transform(features, labels)

    return fs