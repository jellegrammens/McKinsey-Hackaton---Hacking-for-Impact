#import libs
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


def make_model(X_train, Y_train, clf):
    
    clf = clf.fit(X_train, Y_train)
    
    return clf

def make_pred(X_test, clf):
    
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)
    
    
    return y_pred, y_probs

def train_classifier_output_print_results(X_train, Y_train, X_test, Y_test, clf):
    #train model
    clf = make_model(X_train, Y_train, clf)
    
    #evaluate training / test acc
    train_y_pred, train_y_probs = make_pred(X_train, clf)
    test_y_pred, test_y_probs = make_pred(X_test, clf)
    
    
    #print training acc
    acc = np.mean(train_y_pred == Y_train)
    acc_rounded = np.round(acc*100,2)
    print '- TRAIN Accuracy is: ' + str(acc_rounded) 
    
    #print test acc
    acc = np.mean(test_y_pred == Y_test)
    acc_rounded = np.round(acc*100,2)
    print '- TEST Accuracy is: ' + str(acc_rounded)
    
    
    return clf
    
    
    
    
    
    

    
	