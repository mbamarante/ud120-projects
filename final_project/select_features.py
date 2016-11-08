#!/usr/bin/python

import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

def get_features(data_dict, features_number=5):
    features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                     'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                     'shared_receipt_with_poi']

    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    fs = SelectKBest(f_classif, features_number)
    features = fs.fit_transform(features, labels)

    features_scored = dict(zip(features_list, fs.get_support()))

    #features_selected = ['poi', 'salary', 'bonus']
    features_selected = ['poi']
    for feature_scored in features_scored:
        if features_scored[feature_scored] and not feature_scored in features_selected:
            features_selected.append(feature_scored)

    return features_selected