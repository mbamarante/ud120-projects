#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

import pprint

print "data points:", len(enron_data)
print "features:", len(enron_data["SKILLING JEFFREY K"])

poi = 0
for person in enron_data:
    poi += enron_data[person]["poi"]
print "poi:", poi

print enron_data["PRENTICE JAMES"]["total_stock_value"]

pprint.pprint(enron_data["SKILLING JEFFREY K"])

print "Total Payments:"
print "  SKILLING JEFFREY K", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "  LAY KENNETH L     ", enron_data["LAY KENNETH L"]["total_payments"]
print "  FASTOW ANDREW S   ", enron_data["FASTOW ANDREW S"]["total_payments"]

import numpy as np

salary_found = 0
email_found = 0
total_payments_not_found = 0
for person in enron_data:
    if enron_data[person]["email_address"] != "NaN":
        email_found += 1
    if enron_data[person]["salary"] != "NaN":
        salary_found += 1
    if enron_data[person]["total_payments"] == "NaN":
        total_payments_not_found += 1

print "salary found:", salary_found
print "email found:", email_found
print "% payments not found:", total_payments_not_found / float(len(enron_data))