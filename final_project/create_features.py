#!/usr/bin/python

import sys
sys.path.append("../tools/")

def new_features(my_dataset, features_list):
    features_list, my_dataset = paired_emails(my_dataset, features_list)
    features_list, my_dataset = income(my_dataset, features_list)
    return features_list, my_dataset

### pair of emails (2-way communication)
### squared to increase relevance
def paired_emails(my_dataset, features_list):
    for name, data in my_dataset.iteritems():
        if my_dataset[name]["from_poi_to_this_person"] == "NaN" or my_dataset[name]["from_this_person_to_poi"] == "NaN":
            my_dataset[name]["paired_emails"] = "0"
        else:
            my_dataset[name]["paired_emails"] = int(min(
                my_dataset[name]["from_poi_to_this_person"],
                my_dataset[name]["from_this_person_to_poi"]
            ))

    features_list.append("paired_emails")
    return features_list, my_dataset

### total income
def income(my_dataset, features_list):
    for name, data in my_dataset.iteritems():
        if my_dataset[name]["salary"] == "NaN":
            salary = 0
        else:
            salary = my_dataset[name]["salary"]
        if my_dataset[name]["bonus"] == "NaN":
            bonus = 0
        else:
            bonus = my_dataset[name]["bonus"]
        my_dataset[name]["income"] = (salary + bonus)
    features_list.append("income")
    return features_list, my_dataset