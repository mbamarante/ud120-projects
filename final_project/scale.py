#!/usr/bin/python

import sys
sys.path.append("../tools/")

import matplotlib.pyplot
import numpy as np
import pprint
from feature_format import featureFormat, targetFeatureSplit

def do_scale(data_dict, features_list):

    data_array = featureFormat( data_dict, features_list, remove_all_zeroes=False )

    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    data_array = min_max_scaler.fit_transform(data_array)
    # for i in range(1, len(features_list)):
    #     print
    #     #data_array[:, i] = min_max_scaler.fit_transform(data_array[:, i])
    #

    l = 0
    for k in data_dict.keys():
        for c in range(1, len(features_list)):
        #for f in features_list:
            data_dict[k][features_list[c]] = data_array[l, c]
        l += 1

    return data_dict