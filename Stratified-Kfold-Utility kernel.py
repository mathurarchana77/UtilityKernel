# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 23:12:45 2022

@author: Dell
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:05:38 2020

@author: win 10
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:20:06 2020

@author: win 10
"""

import csv
import numpy as np
import pandas as pd
from statistics import mean
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.model_selection import StratifiedKFold

def loadCsv(filename):
    trainSet = []
    testSet = []
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    #print("training set {}".format(dataset[0]))
    for i in range(len(dataset[0])-1):
            for row in dataset:
                    try:
                            row[i] = float(row[i])
                    except ValueError:
                            print("Error with row",i,":",row[i])
                            pass
                    row[-1]=int(float(row[-1]))
    trainSet = dataset        
    return trainSet



def gen_non_lin_separable_data():
    filename = 'S1_standardScaled.csv'

    dataset = pd.read_csv(filename)    
    return dataset




def cobbDKernelone(x1, x2, p=20):
    x1 = x1.flatten()
    x2 = x2.flatten()
    sim = 1.7 + (0.9*(np.dot(x1, x2))**p )
    return sim

def gaussianKernelGramMatrix(X1, X2, K_function=cobbDKernelone):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2)
    return gram_matrix

def training(train, test, fold_no):
    x_train = train.drop(['clas'],axis=1)
    x_train = x_train.values
    #print(x_train)
    y_train = train.clas
    x_test = test.drop(['clas'],axis=1)
    y_test = test.clas
    C=0.1
    clf = svm.SVC(C = C, kernel="precomputed")
    model = clf.fit( gaussianKernelGramMatrix(x_train,x_train), y_train)
        
    p_test = model.predict(gaussianKernelGramMatrix(x_test, x_train))
   
    p_train = model.predict(gaussianKernelGramMatrix(x_train, x_train))
 
    
    from sklearn.metrics import classification_report, confusion_matrix
    
            
    
    
    cm = confusion_matrix(y_test, p_test)
    print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
    #confusionmatrix = np.matrix(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    #print('False Positives\n {}'.format(FP))
    #print('False Negetives\n {}'.format(FN))
    
    #print('True Positives\n {}'.format(TP))
    #print('True Negetives\n {}'.format(TN))
    TPR = TP/(TP+FN)
    #print('Sensitivity \n {}'.format(TPR))
    TNR = TN/(TN+FP)
    #print('Specificity \n {}'.format(TNR))
    Precision = TP/(TP+FP)
    #print('Precision \n {}'.format(Precision))
    Recall = TP/(TP+FN)
    #print('Recall \n {}'.format(Recall))
    Acc = (TP+TN)/(TP+TN+FP+FN)
    print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
    Fscore = 2*(Precision*Recall)/(Precision+Recall)
    print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
    correct = np.sum(p_test == y_test)
    print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
    k=cohen_kappa_score(y_test, p_test)
    print('Çohen Kappa \n{}'.format(k))
    m = matthews_corrcoef(y_test, p_test)
    print('Mathews Correlation Coeff \n{}'.format(m))
    g = (Precision*Recall)**(1/2)
    print('G-measure \n{}'.format(g))

    print('For Fold {} the accuracy is {}'.format(str(fold_no),Acc))

dataset = gen_non_lin_separable_data()
x = dataset
y = dataset.clas
skf = StratifiedKFold(n_splits=10)
fold_no = 1
for train_index,test_index in skf.split(x, y):
      train = dataset.iloc[train_index,:]
      test = dataset.iloc[test_index,:]
      training(train, test, fold_no)
      fold_no += 1


#X1, y1, X2, y2 = gen_non_lin_separable_data()

#X_train, y_train = split_train(X1, y1, X2, y2)
#X_test, y_test = split_test(X1, y1, X2, y2)
#X, y = gen_non_lin_separable_data()
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


