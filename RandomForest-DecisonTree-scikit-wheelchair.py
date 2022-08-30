in# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:22:18 2021

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
from statistics import mean
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn import metrics 
from sklearn.metrics import confusion_matrix


def loadCsv(filename):
    trainSet = []
    testSet = []
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    #print("training set {}".format(dataset))
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
    print("S1_standardScaled --TT- Rondam Forest")
    filename = 'S1_standardScaled_1.csv'

    trainingSet = loadCsv(filename)
    trainingSet=np.array(trainingSet)
    
    X1 = trainingSet[:, 0:22]  
    y1 = [row[-1] for row in trainingSet]    
   
    
    
    return X1, y1

# def gen_non_lin_separable_data():
#     filename = 'train_S2_1.csv'

#     trainingSet = loadCsv(filename)
#     trainingSet=np.array(trainingSet)
    
#     X1 = trainingSet[:, 0:22]  # we only take the first two features.
#     y1 = [int(row[-1]) for row in trainingSet]    
    
#     filename = 'test_S2_1.csv'

#     testSet = loadCsv(filename)
#     testSet=np.array(testSet)
#     X2 = testSet[:, 0:22]  # we only take the first two features.
#     y2 = [int(row[-1]) for row in testSet]    
#     return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):
    X1_train = X1[:50]
    y1_train = y1[:50]
    X2_train = X2[:50]
    y2_train = y2[:50]
    
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    X1_test = X1[50:]
    y1_test = y1[50:]
    X2_test = X2[50:]
    y2_test = y2[50:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test
# def gaussianKernel(x1, x2, sigma=0.1):
#     x1 = x1.flatten()
#     x2 = x2.flatten()
#     sim = np.exp(- np.sum( np.power((x1 - x2),2) ) /float( 2*(sigma**2) ) )
#     return sim

# def polyKernel(x1, x2, p=2):
#     x1 = x1.flatten()
#     x2 = x2.flatten()
#     sim = (1 + np.dot(x1, x2)) ** p
#     return sim

# def cobbDKernel(x1, x2, p=4):
#     x1 = x1.flatten()
#     x2 = x2.flatten()
#     sim = 1/(1 -  (np.dot(x1, x2) ** p))
#     return sim

# def cobbDKernelone(x1, x2, p=7):
#     x1 = x1.flatten()
#     x2 = x2.flatten()
#     sim =7*(np.dot(x1, x2))**p 
#     return sim

# def gaussianKernelGramMatrix(X1, X2, K_function=cobbDKernelone):
#     gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
#     for i, x1 in enumerate(X1):
#         for j, x2 in enumerate(X2):
#             gram_matrix[i, j] = K_function(x1, x2)
#     return gram_matrix



X, y = gen_non_lin_separable_data()
#X_train, y_train = split_train(X1, y1, X2, y2)
#X_test, y_test = split_test(X1, y1, X2, y2)

# C=0.1
# clf = svm.SVC(C = C, kernel="precomputed")
# model = clf.fit( gaussianKernelGramMatrix(X_train,X_train), y_train)
# p = model.predict( gaussianKernelGramMatrix(X_test, X_train))
# print(p)


# from sklearn.svm import SVC
# svclassifier = SVC(kernel='rbf')
# svclassifier.fit(X_train, y_train)

# from sklearn.svm import SVC
# svclassifier = SVC(kernel='poly', degree=3)
# svclassifier.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#RANDOM Forest
clf = RandomForestClassifier(n_estimators = 3)
 
# Train the model using the training sets
#clf.fit(X_train, y_train)

#DECISION TREE
#clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 9, min_samples_leaf = 8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

#print(classification_report(y_train, y_pred_train))
#print("ACCURACY OF THE MODEL: train", metrics.accuracy_score(y_train, y_pred_train))

#print(classification_report(y_test, y_pred_test))
#print("ACCURACY OF THE MODEL: test", metrics.accuracy_score(y_test, y_pred_test))



k=cohen_kappa_score(y_train, y_pred_train)
#print('Çohen Kappa \n{}'.format(k))
m = matthews_corrcoef(y_train, y_pred_train)
#print('Mathews Correlation Coeff \n{}'.format(m))
#g = (Precision*Recall)**(1/2)
#print('G-measure \n{}'.format(g))
#f = f1_score(y_test, y_pred_test, average='micro')
#print('F1 score \n{}'.format(f))
cm = confusion_matrix(y_train, y_pred_train)
#print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
#print('False Positives\n {}'.format(FP))
#print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
#print('Sensitivity \n {}'.format(TPR))
# TNR = TN/(TN+FP)
# #print('Specificity \n {}'.format(TNR))
Precision = TP/(TP+FP)
print('Precision \n {}'.format(mean(Precision)))
Recall = TP/(TP+FN)
print('Recall \n {}'.format(mean(Recall)))
Acc = (TP+TN)/(TP+TN+FP+FN)
print('Average Train Acc \n{}'.format(mean(Acc)))
Fscore = 2*(Precision*Recall)/(Precision+Recall)
print('Average Train Fscore \n{}'.format(mean(Fscore)))
# # correct = np.sum(p_train == y_train)
# # print("Gaussian k0=0.1, p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
# # k=cohen_kappa_score(y_test, p_test)
# # print('Çohen Kappa \n{}'.format(k))
# # m = matthews_corrcoef(y_test, p_test)
# # print('Mathews Correlation Coeff \n{}'.format(m))
# # g = (Precision*Recall)**(1/2)
# # print('G-measure \n{}'.format(g))

# # j = jaccard_score(y_test, p_test, average='weighted')
# # print('Jaccard Index \n{}'.format(j))
print('***************')
# # with open('result.csv','a') as fd:
    
# #     fd.write("Training Accuracy,")
# #     writer = csv.writer(fd)
# #     writer.writerow(Acc)
# #     fd.write("Avg Training Accuracy,")
# #     writer.writerow([mean(Acc)])    
# #     fd.write("Training Fscore,")
# #     writer.writerow(Fscore)
# #     fd.write("Avg Training Fscore,")
# #     writer.writerow([mean(Fscore)])


cm = confusion_matrix(y_test, y_pred_test)
#print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
# #confusionmatrix = np.matrix(cm)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
# #print('False Positives\n {}'.format(FP))
# #print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
# #print('Sensitivity \n {}'.format(TPR))
# TNR = TN/(TN+FP)
# #print('Specificity \n {}'.format(TNR))
Precision = TP/(TP+FP)
print('\n\nPrecision \n{}'.format(mean(Precision)))
Recall = TP/(TP+FN)
print('Recall \n{}'.format(mean(Recall)))
Acc = (TP+TN)/(TP+TN+FP+FN)
print(' Average Test Acc \n{}'.format(mean(Acc)))
Fscore = 2*(Precision*Recall)/(Precision+Recall)
print('Average Test Fscore \n{}'.format(mean(Fscore)))
# correct = np.sum(p_test == y_test)
# print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
k=cohen_kappa_score(y_test, y_pred_test)
print('Çohen Kappa \n{}'.format(k))
m = matthews_corrcoef(y_test, y_pred_test)
print('Mathews Correlation Coeff \n{}'.format(m))
# g = (Precision*Recall)**(1/2)
# print('G-measure \n{}  {}'.format(g, np.mean(g)))

# j = jaccard_score(y_test, p_test, average='weighted')
# print('Jaccard Index \n{}'.format(j))
print('***************')
# # with open('result.csv','a') as fd:

# #     fd.write("Testing Accuracy,")
# #     writer = csv.writer(fd)
# #     writer.writerow(Acc)
# #     fd.write("Avg Testing Accuracy,")
# #     writer.writerow([mean(Acc)])    
# #     fd.write("Testing Fscore,")
# #     writer.writerow(Fscore)
# #     fd.write("Avg Testing Fscore,")
# #     writer.writerow([mean(Fscore)])




############################################################################
# LOOCV - Leave one out CV

