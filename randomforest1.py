import numpy as np
import csv
import re


whitelist = {}

def output_predictions(filename,testingdata, classf):
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        for item in classf.predict(testingdata):
            spamwriter.writerow([item])

def import_whitelist(filename):
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            whitelist[row[1]] = row[0]
                
def test_whitelist(whitelist, word):
    if word in whitelist:
        return True
    for entry in whitelist:
        if whitelist[entry] == 'contains':
            if entry in word:
                return True
            else:
                continue
        elif whitelist[entry] == 'regex search':
            if re.search(entry, word) != None:
                return True
            else:
                continue
    return False


import_whitelist('whitelist.csv')

def import_fin_files(filename, whitelist):
    """ this function takes a financial file and a whitelist and outputs a list containing a text concatenation of white-listed verified words, 1 item per line"""
    with open('TrainingSetFull.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        headers = spamreader.next()
        j = 0
        for cell in headers:
            if cell == 'Operating Status':
                coding_count = j
            else:
                j += 1
        fin_file = []
        operating_list = []
        coding_targets = []
        for row in spamreader:
            string_list = []
            coding_targets.append(row[40])
            operating_list.append(row[coding_count])
            for i in xrange(1,coding_count):
                for word in row[i].lower().split():
                    if test_whitelist(whitelist, word):
                        string_list.append(word)
            fin_file.append(' '.join(string_list))
        return fin_file, coding_targets, operating_list

#fin, targets, operating = import_fin_files('TrainingSetFull.csv',whitelist)

def save_file(filename, data):
    """saves data to a file for quicker retrieval later.  Best for fin_file"""
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        for item in data:
            spamwriter.writerow([item])

#save_file('fin_file.csv',fin)
#save_file('op_file.csv',operating)
#save_file('targets.csv',targets)

def load_file(filename):
    """ loads data from file if processing takes a long time"""
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        data = []
        for row in spamreader:
            data.append(row[0])
    return data

fin = load_file('fin_file.csv')
targets = load_file('targets.csv')
operating = load_file('op_file.csv')

#Set parameters
StartRow = 1
StartTest = 12000
EndTest = 13000

train_file = [fin[x] for x in xrange(StartRow, StartTest) if operating[x] == 'PreK-12 Operating']
train_targets = [targets[x] for x in xrange(StartRow, StartTest) if operating[x] == 'PreK-12 Operating']
test_file = [fin[x] for x in xrange(StartTest, EndTest)]


#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1,charset_error='ignore')

X_train = vectorizer.fit_transform(train_file)
X_train = X_train.todense()

#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
#clf = MultinomialNB()
#clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)
clf = RandomForestClassifier(n_estimators = 500, compute_importances=True)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit_transform(X_train, train_targets)

testdata = vectorizer.transform(test_file)
testdata = testdata.todense()


output_predictions('predictions.csv',testdata, clf)
