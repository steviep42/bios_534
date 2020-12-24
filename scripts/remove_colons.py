#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is to remove the stupid colons used in the data files
referenced on the following site where for some reason they have the 
column number prefixed onto the feature instead of using column
names like the rest of the civilized data science world

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

@author: Steve 
"""

## Read in files
lines = []
fname = "/Users/esteban/Downloads/cod-rna.txt"
file_object  = open(fname) 
data = file_object.readlines()

newlist = []
for line in data:
    words = line.split()
    
    for word in words:
        spw = word.split(":")
        if (len(spw) == 1):
            newlist.append(spw[0])
        else:
            newlist.append(spw[1])
    lines.append(newlist)
    newlist = []

MyFile=open('/Users/esteban/Downloads/cod_rna_train.txt','w')

for element in lines:
     listToStr = ','.join(map(str, element))
     MyFile.write(listToStr)
     MyFile.write('\n')
MyFile.close()

import pandas as pd
colnames = ['ncrna','deltag','seqlen','afreq1','ufreq1','cfreq1','afreq2','ufreq2','cfreq2']
df = pd.read_csv("/Users/esteban/Downloads/cod_rna_train.txt",names=colnames)

# Let's map -1 (a non coding region) to 0
# and 1 (a coding region) to 1

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

le = LabelBinarizer().fit(df.ncrna)
print(le.classes_)
df.ncrna = le.transform(df.ncrna)

# 
X = df.drop('ncrna',axis=1)
y = df.ncrna

Standardisation = StandardScaler() 
X_scaled= Standardisation.fit_transform(X) 

# 
lin_clf = SVC()
lin_clf.fit(X_scaled, y)
lin_clf.score(X_scaled,y).round(2)

svc_preds = lin_clf.predict(X_scaled)
print(classification_report(y, svc_preds))


# Regression
logit_model = LogisticRegression(solver='liblinear', random_state=0)
logit_model.fit(X_scaled,y)
logit_model.score(X_scaled,y).round(2)

#

logit_preds = logit_model.predict(X_scaled)
print(classification_report(y, logit_preds))

#

def predictor(model,X,y_actual):
    from sklearn import metrics
    perf_measure = {}
    preds = model.predict(X)
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_actual, preds).ravel()
    perf_measure['sensitivity'] = round(tp/(tp+fn),2)
    perf_measure['specificity'] = round(tn/(tn+fp),2)
    perf_measure['precision']  = round(tp/(tp+fp),2)
    perf_measure['accuracy']   = round((tn+tp)/(tn+fp+fn+tp),2)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, preds)
    perf_measure['auc']  = round(metrics.auc(fpr, tpr),2)
    return(perf_measure)
    
 
predictor(logit_model,X_scaled,y)



