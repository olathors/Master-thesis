import math
from itertools import combinations

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from sklearn.metrics import roc_auc_score,roc_curve,auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

#from de_long_evaluation import delong_roc_test



def compute_metrics_binary(y_true, y_pred_proba,threshold = 0.5,verbose=0):
    '''
    
    Compute the following metrics for a binary classification problem: 
    AUC, Accuracy, F1 Score, Precision, Recall and Confusion matrix.

    Parameters
    ----------
    y_true: list or array containing the true values.

    y_pred_proba: array containing the predicted scores

    threshold: cutoff point to encode the predicted scores. 1 if score >= threshold, else 0.

    verbose: Flag to print out results.

    Returns
    ---------

    Dict with metrics and their values.

    '''
    

    y_pred_proba = torch.softmax(y_pred_proba, 1)

    y_pred_proba = get_numpy_array(y_pred_proba)
    #y_pred_label = y_pred_proba.copy()
    #y_pred_label[y_pred_proba >= threshold] = 1
    #y_pred_label[y_pred_proba < threshold] = 0

    y_pred_label = np.argmax(y_pred_proba, 1)

    y_true = get_numpy_array(y_true)


    auc = roc_auc_score(y_true, y_pred_proba, labels = [0,1,2], multi_class="ovr", average= 'weighted')
    accuracy = accuracy_score(y_true, y_pred_label)
    f1score = f1_score(y_true, y_pred_label, labels = [0,1,2], average = 'weighted')
    recall = recall_score(y_true, y_pred_label, labels = [0,1,2], average = 'weighted')
    precision = precision_score(y_true, y_pred_label, labels = [0,1,2], average = 'weighted')
    conf_mat = confusion_matrix(y_true, y_pred_label, labels = [0,1,2])

    if verbose > 0:
        print('----------------')
        print("Total samples in batch:",y_true.shape)
        print("AUC:       %1.3f" % auc)
        print("Accuracy:  %1.3f" % accuracy)
        print("F1:        %1.3f" % f1score)
        print("Precision: %1.3f" % precision)
        print("Recall:    %1.3f" % recall)
        print("Confusion Matrix: \n", conf_mat)
        print('----------------')
    metrics = {
        'auc':auc,
        'accuracy':accuracy,
        'f1score':f1score,
        'precision':precision,
        'recall':recall,
        'conf_mat':conf_mat
    }

    
    return metrics

def get_numpy_array(arr):
    if isinstance(arr,torch.Tensor):
        return arr.cpu().detach().numpy()
    elif isinstance(arr,list):
        return np.array(arr)
    return arr