'''
This file contains several functions to evaluate a supervised learning model's performance on several metrics and plots.
'''

#import libs / functions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from scipy import interp

##################
# CLASSIFICATION #
##################

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    
    Parameters
    ----------
    - cm: confusion matrix object returned from the confusion_matrix function from sklearn
    - classes: np array / list with real names of classes 
    - normalise: boolean, normalizes the numbers if True
    - title: title for plot
    - cmap: type of colormap you want for the confusion matrix
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #if classes is None:
        #classes = range(0, cm.shape[0])
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def calculate_confusion_matrix_numbers(cm):
    '''
    Function that calculates confusion matrix numbers and rates.
    
    Parameters
    ----------
    - cm: confusion matrix object returned from the confusion_matrix function from sklearn

    Returns
    -------
    - TP, FN, TN, FP, TPR, FNR, TNR, FPR
    '''

    #binary
    if cm.shape[0] == 2:
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]

    #multi-class
    else:
        FP = cm.sum(axis=0) - np.diag(cm) 
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = np.round(TP.astype(float)/(TP+FN) * 100, 2)
    # Specificity or true negative rate
    TNR = np.round(TN.astype(float)/(TN+FP) * 100, 2)
    # Fall out or false positive rate
    FPR = np.round(FP.astype(float)/(FP+TN) * 100, 2)
    # False negative rate
    FNR = np.round(FN.astype(float)/(TP+FN) * 100, 2)
    
    return TP, FN, TN, FP, TPR, FNR, TNR, FPR

def plot_roc_curve(y_real, y_probs, class_names = None):
    '''
    Function that plots the ROC-Curve. If multi-class it will plot all the roc-curves on the same plot

    Parameters
    ----------
    - y_real: np array with real y-values
    - y_probs: np array with probability estimates
    - class_names: np array or list with real class_names to plot on confusion matrix (instead of 0, 1, ...)
    '''

    n_classes = len(np.unique(y_real))
    lw = 2

    #binary
    if n_classes == 2:
        fpr, tpr, thresholds = roc_curve(y_real, y_probs[:,1])
        roc_auc = auc(fpr,tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    if n_classes > 2:
        y_real = label_binarize(y_real, np.unique(y_real))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_real[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(8,8))
        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            class_name = class_names[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class ' + class_name + ' (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


def evaluate_supervised_learning_model_classification(y_real, y_pred, y_probs, class_names = None):
    '''
    Function that prints an evaluation report of a classifier's (binary / multi-class) performance on a dataset:
    - accuracy
    - confusion matrix plot + TPR, FNR, TNR, FPR
    - roc curve

    Parameters
    ----------
    - y_real: np array with real y-values
    - y_pred: np array with predicited y-values from classifier
    - y_probs: np array with probability estimates
    - class_names: np array or list with real class_names to plot on confusion matrix (instead of 0, 1, ...)

    Returns
    -------
    a dictionary with 2 np arrays with row numbers of (in)correctly classified observations to see where your classifier failed
    '''

    #make class_names that are y_real value names if class_names is None
    if class_names is None:
        class_names = np.array(range(0, len(np.unique(y_real)))).astype(str)
    
    #accuracy
    acc = np.mean(y_pred == y_real)
    acc_rounded = np.round(acc*100,2)
    print '- Accuracy is: ' + str(acc_rounded) + '% \n'

    #confusion matrix - plot
    cm = confusion_matrix(y_real, y_pred) 
    cm_title = 'Confusion Matrix:' 
    print cm_title
    print '-' * len(cm_title)
    plt.figure()
    plot_confusion_matrix(cm, class_names, normalize=False, title='CM')
    plt.figure()
    plot_confusion_matrix(cm, class_names, normalize=True, title='CM (normalised)')    
    plt.show()
    
    #confusion matrix - rates in percentages (TPR, FPR, TNR, FNR)
    (TP, FN, TN, FP, TPR, FNR, TNR, FPR) = calculate_confusion_matrix_numbers(cm)

    cm_numbers_title = 'Confusion Matrix Numbers and Rates:'
    print cm_numbers_title
    print '-' * len(cm_numbers_title)

    #check how many classes there are present
    number_of_classes = len(np.unique(y_real))
    
    #multi-class
    if number_of_classes > 2:
        c = 0
        for class_name in class_names:
            print 'Class: ' + class_name
            print '---'
            print 'TP: ' + str(TP[c])
            print 'FN: ' + str(FN[c]) 
            print 'TN: ' + str(TN[c]) 
            print 'FP: ' + str(FP[c]) 
            print 'TPR: ' + str(TPR[c]) + ' %'
            print 'FNR: ' + str(FNR[c]) + ' %'
            print 'TNR: ' + str(TNR[c]) + ' %'
            print 'FPR: ' + str(FPR[c]) + ' % \n'
            c += 1

    #binary class
    else:
        print 'TP: ' + str(TP) 
        print 'FN: ' + str(FN) 
        print 'TN: ' + str(TN)
        print 'FP: ' + str(FP) 
        print 'TPR: ' + str(TPR) + ' %'
        print 'FNR: ' + str(FNR) + ' %'
        print 'TNR: ' + str(TNR) + ' %'
        print 'FPR: ' + str(FPR) + ' % \n'
        
    #precision, recall, f1, support (number of observations per class)
    prf1_title = 'Precision, Recall and F1-score:'
    print prf1_title
    print '-' * len(prf1_title)
    print classification_report(y_real, y_pred)

    #roc curve
    roc_curve_title = 'ROC-Curve:'
    print roc_curve_title 
    print '-' * len(roc_curve_title) + '\n'
    plot_roc_curve(y_real, y_probs, class_names = class_names)
    
    #make dictionary where we keep observation numbers of (in)correctly classified labels
    dct = {}
    
    #observations correctly classified
    dct['class_right'] = np.where(y_real == y_pred)[0]
    
    #observations incorrectly classified
    dct['class_wrong'] = np.where(y_real != y_pred)[0]
    
    return dct


def plot_label_with_two_numerical_features(x, y, features, y_labels = None, obs_to_plot = None, plot_both = False):
    '''
    Function that plots a scatter plot with two numerical variables on the axis with each class in a distinct color.
    Optional parameter to plot only a subset of observations. 
    For example: those that are wrongly classified. They will appear in a red color.
    
    Parameters
    ----------
    - x: np array / pandas df with features
    - y: np array / pandas series with true labels
    - features: list with features names or indexes to plot
    - y_labels: list with label names of classes (y)
    - obs_to_plot: np array or list with observation numbers to only plot: color will be red
    - plot_both: boolean, if True: plot all observations
    '''
    
    #vector with colors
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
    
    #save each feature in different variable
    var1 = features[0]
    var2 = features[1]
    
    #x: transform pd df to numpy array
    if isinstance(x, pd.DataFrame):
        x = x[features].as_matrix()
        var1 = 0
        var2 = 1
        
    #y: transform list to numpy array
    if isinstance(y, list):
        y = np.array(y)
    
    #split observations of obs_to_plot is not None
    if obs_to_plot is not None:
        x_sub = x[obs_to_plot].copy()
        x = np.delete(x, obs_to_plot, 0)
        y_sub = y[obs_to_plot].copy()
        y = np.delete(y, obs_to_plot, 0)
    
    #if y_labels is None, take y values as y_labels
    if y_labels is None:
        y_labels = np.array(range(0,len(np.unique(y)))).astype(str)   
        
    if obs_to_plot is None:
        for n, y_class in enumerate(np.unique(y)):
            idx = np.where(y == y_class)[0]
            plt.scatter(x[idx, var1], x[idx, var2], color=colors[n], label=y_labels[n])
    else:
        if plot_both:
            for n, y_class in enumerate(np.unique(y)):
                import pdb
                idx = np.where(y == y_class)[0]
                #pdb.set_trace()
                plt.scatter(x[idx, var1], x[idx, var2], color=colors[n], label=y_labels[n])         
        plt.scatter(x_sub[:, var1], x_sub[:, var2], color='r', label='Misclassified')
        
    plt.xlabel(features[0])
    plt.ylabel(features[1])      
    plt.legend(loc=0)
    plt.title("Scatter Plot")
    plt.show()


##############
# REGRESSION #
##############


def evaluate_supervised_learning_model_regression(y_real, y_pred, info = True):
    expl_var_title = 'Explained variance score: '
    mae_title = 'Mean absolute error: '
    mse_title = 'Mean squared error: '
    msle_title = 'Mean squared logarithmic error: '
    medae_title = 'Median absolute error: '
    r2_title = 'R2 score: '

    #explained variance
    expl_var = explained_variance_score(y_real, y_pred)
    print expl_var_title + str(expl_var)

    #r2
    r2 = r2_score(y_real, y_pred)
    print r2_title + str(r2)

    #mean squared error
    mse = mean_squared_error(y_real, y_pred)
    print mse_title + str(mse)

    #mean squared log error
    try:
        msle = mean_squared_log_error(y_real, y_pred)
        print msle_title + str(msle)
    except ValueError:
        print 'Mean Squared Logarithmic Error: failed, cannot be used when targets contain negative values.'

    #mean absolute error
    mae = mean_absolute_error(y_real, y_pred)
    print mae_title + str(mae)

    #median absolute error
    medae = median_absolute_error(y_real, y_pred)
    print medae_title + str(medae)




