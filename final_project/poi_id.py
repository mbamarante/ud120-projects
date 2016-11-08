#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from select_features import get_features
from select_features import print_features_score
from remove_outliers import clean_outilers
from scale import do_scale
from create_features import new_features
from create_classifier import new_classifier, new_tunned_classifier
from my_validate import do_validate

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

raw_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                     'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                     'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                     'from_this_person_to_poi', 'shared_receipt_with_poi']

### task 2: remove outliers
my_dataset = clean_outilers(data_dict, raw_features_list, percentile=5)

### task 3: create new feature(s)
raw_features_list, my_dataset = new_features(my_dataset, raw_features_list)

## scale
my_dataset = do_scale(my_dataset, raw_features_list)

## feature selection
print_features_score(my_dataset, raw_features_list)
features_list = ['poi', 'exercised_stock_options', 'income', 'deferred_income', 'long_term_incentive']

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = False)
# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# clf = new_classifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = new_tunned_classifier(my_dataset, features_list)

do_validate(clf, 10, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)