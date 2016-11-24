import sys
sys.path.append("../tools/")

from feature_format import featureFormat

def counting_zeros_do(data, features_list, feature):
    total_count = 0
    count = 0
    feature_data = data[:,features_list.index(feature)]
    for i in feature_data:
        total_count += 1
        if i == 0:
            count += 1
    if (int(float(count)*100/float(total_count)) > 60):
        print feature, int(float(count)*100/float(total_count))

def counting_zeros(data, features_list):
    data = featureFormat(data, features_list)
    for i in features_list[:-1]:
        counting_zeros_do(data, features_list, i)

def only_zeros(data, features_list):
    for rec in data:
        zero = 0
        count = 0
        for feat in features_list:
            count += 1
            if (data[rec][feat] == 'NaN' or data[rec][feat] == 0):
                zero += 1

        if (zero == count):
            print rec