#!/usr/bin/python

import sys
sys.path.append("../tools/")

import matplotlib.pyplot
import numpy as np
import pprint
from feature_format import featureFormat, targetFeatureSplit

def show_boxplot(data, name):
    import seaborn as sns
    sns.violinplot(data, color="green", orient="v", title=name)
    sns.plt.show()

def show_features(data_dict, feat1, feat2):
    data = featureFormat(data_dict, [feat1, feat2])
    data_poi = featureFormat(data_dict, ["poi"])
    for point in data:
        matplotlib.pyplot.scatter(point[0], point[1])

    matplotlib.pyplot.xlabel(feat1)
    matplotlib.pyplot.ylabel(feat2)
    matplotlib.pyplot.show()
    # print "highest", feat1, dict_find(feat1, data[:, 0].max())
    # print "highest", feat2, dict_find(feat2, data[:, 1].max())

# def dict_find(key, value):
#     for name, data in data_dict.iteritems():
#         if (data[key] == value):
#             return name

def get_higher(data, percentil):
    return np.argsort(data)[::-1][0:int(len(data) * percentil)]

def get_lower(data, percentil):
    return np.argsort(data)[0:int(len(data) * percentil)]

def clean_outilers(data_dict, features_list, percentile=10):
    data_dict.pop("TOTAL", 0)
    data_array = featureFormat( data_dict, features_list, remove_all_zeroes=False )
    to_remove = []

    for i in range(1, len(features_list)):
        for h in get_higher(data_array[:, i], percentile/100):
            if data_array[h, i] > np.percentile(data_array[:, i], 75 and data_array[h, 0] == 0.0): # is not poi
                to_remove.append(data_dict.keys()[h])
        for l in get_lower(data_array[:, i], percentile / 100):
            if data_array[h, i] < np.percentile(data_array[:, i], 25 and data_array[h, 0] == 0.0):  # is not poi
                to_remove.append(data_dict.keys()[h])

    for r in to_remove:
        data_dict.pop(r, 0)

    return data_dict