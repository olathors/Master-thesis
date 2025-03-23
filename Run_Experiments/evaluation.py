import math
from itertools import combinations
from itertools import cycle

import sklearn.preprocessing
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erfcinv
import sklearn
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score

#from de_long_evaluation import delong_roc_test



def compute_metrics_binary(y_true, y_pred_proba, classes, class_id, threshold = 0.5, verbose=0):
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
    labels = list(range(0, classes))

    y_pred_proba = torch.softmax(y_pred_proba, 1)

    y_pred_proba = get_numpy_array(y_pred_proba)
    #y_pred_label = y_pred_proba.copy()
    #y_pred_label[y_pred_proba >= threshold] = 1
    #y_pred_label[y_pred_proba < threshold] = 0

    y_true = get_numpy_array(y_true)

    if classes == 2:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1], labels = labels, average= 'macro')
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        _,_, optimal_threshold = find_optimal_cutoff(fpr, tpr, thresholds)
        y_pred_label = y_pred_proba.copy()
        y_pred_label[y_pred_proba >= optimal_threshold] = 1
        y_pred_label[y_pred_proba < optimal_threshold] = 0
        y_pred_label = np.argmax(y_pred_label, 1)
        #roc_auc = sklearn.metrics.auc(fpr, tpr)
        #best_roc_curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot().figure_
    else:
        auc = roc_auc_score(y_true, y_pred_proba, labels = labels, multi_class="ovr", average= 'macro')
        y_pred_label = np.argmax(y_pred_proba, 1)
        optimal_threshold = threshold
        #best_roc_curve = multiclass_curve(y_true, y_pred_proba, classes, class_id)


    accuracy = accuracy_score(y_true, y_pred_label)
    f1score = f1_score(y_true, y_pred_label, labels = labels, average= 'macro')
    recall = recall_score(y_true, y_pred_label, labels = labels, average= 'macro')
    precision = precision_score(y_true, y_pred_label, labels = labels, average= 'macro', zero_division = 0)
    conf_mat = confusion_matrix(y_true, y_pred_label, labels = labels)

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
        'conf_mat':conf_mat,
        'roc_auc': [y_true, y_pred_proba, optimal_threshold]
    }

    
    return metrics

def get_numpy_array(arr):
    if isinstance(arr,torch.Tensor):
        return arr.cpu().detach().numpy()
    elif isinstance(arr,list):
        return np.array(arr)
    return arr

def multiclass_curve(y_true, y_score, n_classes, target_names):
    #This function is from sklearn website.

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    y_onehot_test = label_binarizer.fit_transform(y_true)   

    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.3f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "navy"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 1),
            #despine=True,
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )

    return fig

def find_optimal_cutoff(fpr, tpr, thresholds):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate.
    
    Parameters
    ----------
    fpr: False positive rate

    tpr : True positive rate

    Returns
    -------
    cutoff value

    """
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))  # Minimum distance to the upper left corner (By Pathagoras' theorem)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    return optimal_sensitivity, optimal_specificity, optimal_threshold