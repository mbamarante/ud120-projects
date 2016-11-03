#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

data_dict.pop("TOTAL", 0)

salary_list = list()

for key, value in data_dict.iteritems():
    salary_list.append([key, value["salary"]])

import pprint
pprint.pprint(sorted(salary_list, key=lambda tup: tup[1]))
#     print("key:", key)
#     print("value:", value)
#
# for person in data_dict:
#     salary = data_dict[person]["salary"]
#     bonus = data_dict[person]["bonus"]
#     #below, we need to ignore NaN values which were removed in the data array but not in
#     #the dictionary
#     if salary != 'NaN' and bonus != 'NaN' and salary >= 5000000 and bonus > 1000000:
#         print(person)

from operator import itemgetter
#print sorted(data_dict, key=itemgetter('salary'))
#print sorted(data_dict, key=lambda k: k["salary"])

data = featureFormat(data_dict, features)


### your code below



for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print data[:,0].max()

26,704,229.0