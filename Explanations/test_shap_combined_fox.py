import sys
import os
sys.path.append('../')
import shap
import torch
import cv2
from torch.utils.data import DataLoader
from mri_dataset_combined_shap import MRI_Dataset, MRI_Dataset_combined
from torchvision.models.efficientnet import efficientnet_v2_l, efficientnet_v2_m
import numpy 
import torchvision
from combined_classifier_shap import *
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True
from custom_image_shap import image as image_plot
import seaborn as sns
import pandas as pd
from scipy.special import softmax
from datetime import datetime


def run(label_wanted, label_true, num_images, num_samples, plot_path):

    experiment_name = 'experiment_predlabel-'+str(label_wanted)+'_truelabel-'+str(label_true)+'_numimages-'+str(num_images)+'_numsamples-'+str(num_samples)+'_time-'+datetime.now().strftime("%Y%m%d%H%M")

    plot_path = os.path.join(plot_path, experiment_name)

    if not os.path.exists(plot_path ):
        os.makedirs(plot_path)

    plt.style.use('seaborn-v0_8') 

    device = torch.device('cpu')

    #Initializing datasets
    train_dataset = MRI_Dataset_combined('/Users/olath/Documents/GitHub/Master-thesis/Datasets/train-CN-sMCI-pMCI-AD','/Users/olath/Documents/ADNI_SLICED_RESCALED/', device, slice= [12, 72, 43, 6, 58, 43])
    val_dataset = MRI_Dataset_combined('/Users/olath/Documents/GitHub/Master-thesis/Datasets/val-CN-sMCI-pMCI-AD','/Users/olath/Documents/ADNI_SLICED_RESCALED/', device, slice= [12, 72, 43, 6, 58, 43])

    train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)
    validation_loader  = DataLoader(val_dataset, batch_size=1071, shuffle=True)

    #Initializing models
    axial1 = efficientnet_v2_m(num_classes = 3)
    sagittal1 = efficientnet_v2_m(num_classes = 3)
    coronal1 = efficientnet_v2_m(num_classes = 3)
    axial2 = efficientnet_v2_m(num_classes = 2)
    sagittal2 = efficientnet_v2_m(num_classes = 2)
    coronal2 = efficientnet_v2_m(num_classes = 2)

    model = DoubleCombinedClassifierLogReg(4, axial1, sagittal1, coronal1, axial2, sagittal2, coronal2, dropout = 0.4, num_outputs = 250)
    model.load_state_dict(torch.load('/Users/olath/Downloads/model_4classDouble_202503171605_best.pth', weights_only=True, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    X, y = next(iter(validation_loader))

    best_pred = 0
    best_pred_index = 0

    best = list()
    scores = list()
    best_dict = {}

    for j in range (0, 1):
        out = model(X[0], X[1], X[2], X[3], X[4], X[5])
        for i in range(0, len(out)):
            if y[i] == label_true and (out[i].max() == out[i][label_wanted]):
                best_dict[out[i][label_wanted].cpu().detach().item()] = i
                scores.append(out[i][label_wanted].cpu().detach().item())
                if out[i][label_wanted] > best_pred:
                    best_pred = out[i][label_wanted]
                    best_pred_index = i
        

    scores.sort()
        
    for score in scores:
        best.append([X[0][best_dict[score]].cpu().detach(), X[1][best_dict[score]].cpu().detach(), X[2][best_dict[score]].cpu().detach(), 
                X[3][best_dict[score]].cpu().detach(), X[4][best_dict[score]].cpu().detach(), X[5][best_dict[score]].cpu().detach()])

    best_pred = best_pred.cpu().detach().numpy().round(4)

    y_num = y[best_pred_index:best_pred_index+1].numpy()

    just_labels = {0: 'CN', 1:'sMCI',2:'pMCI',3:'AD'}

    labels = {0: 'True class: CN\nPredicted: ' +str(just_labels[label_wanted]), 1:'True class: sMCI\nPredicted: ' +str(just_labels[label_wanted]), 2: 'True class: pMCI\nPredicted: ' +str(just_labels[label_wanted]), 3:'True class: AD\nPredicted: ' +str(just_labels[label_wanted])}
    true_labels = list()

    for  label in y_num:    

        true_labels.append(labels[label])

    #Setting up and running the explainer
    X_train, y_train = next(iter(train_loader))

    best = np.array(best)

    best_input = np.transpose(best[-num_images:], (1, 0, 2, 3, 4)).tolist()

    for i in range(0, len(best_input)):
        best_input[i] = torch.tensor(best_input[i]).to(device)
        
    explainer = shap.GradientExplainer(model, X_train)

    shap_values = explainer.shap_values(best_input, nsamples = num_samples)

    plot_pred = model(X[0][best_pred_index:best_pred_index+1], X[1][best_pred_index:best_pred_index+1], X[2][best_pred_index:best_pred_index+1], 
            X[3][best_pred_index:best_pred_index+1], X[4][best_pred_index:best_pred_index+1], X[5][best_pred_index:best_pred_index+1])

    plot_pred = plot_pred.cpu().detach().numpy()

    plot_pred = plot_pred[0].round(4)

    mean_pred = model(best_input[0],best_input[1],best_input[2],best_input[3],best_input[4],best_input[5]).detach().cpu().numpy()

    pred_means = [mean_pred[:,0].mean(),mean_pred[:,1].mean(),mean_pred[:,2].mean(),mean_pred[:,3].mean()]

    plot_shap_images(X, shap_values, true_labels, plot_pred, plot_path)

    plot_bargraph(shap_values, plot_pred, label_wanted, label_true, plot_path, image_index = -1, fig_size = (8,8), palette = None)

    plot_stripplot(shap_values, pred_means, label_wanted, label_true, plot_path, num_images = num_images, fig_size = (8,8), palette = None)

    plot_extraplots(shap_values, label_wanted, label_true, plot_path, num_images = num_images, fig_size = (8,18))


def plot_shap_images(X, shap_values, true_labels,  plot_pred, plot_path):

    display_img1 = torch.permute(X[0][0].cpu(), (1, 2, 0)).numpy()
    display_img2 = torch.permute(X[1][0].cpu(), (1, 2, 0)).numpy()
    display_img3 = torch.permute(X[2][0].cpu(), (1, 2, 0)).numpy()
    display_img4 = torch.permute(X[3][0].cpu(), (1, 2, 0)).numpy()
    display_img5 = torch.permute(X[4][0].cpu(), (1, 2, 0)).numpy()
    display_img6 = torch.permute(X[5][0].cpu(), (1, 2, 0)).numpy()

    display_val1 = np.transpose(shap_values[0], (4, 0, 2, 3, 1))
    display_val2 = np.transpose(shap_values[1], (4, 0, 2, 3, 1))
    display_val3 = np.transpose(shap_values[2], (4, 0, 2, 3, 1))
    display_val4 = np.transpose(shap_values[3], (4, 0, 2, 3, 1))
    display_val5 = np.transpose(shap_values[4], (4, 0, 2, 3, 1))
    display_val6 = np.transpose(shap_values[5], (4, 0, 2, 3, 1))
    
    image_plot([display_val1[i][-1] for i in range(4)], display_img1, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Axial - Slice: 12 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))

    experiment_tag = '_'.join(['shap_image_input1', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val2[i][-1] for i in range(4)], display_img2, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Sagittal - Slice: 72 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input2', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val3[i][-1] for i in range(4)], display_img3, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Coronal - Slice: 43 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input3', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val4[i][-1] for i in range(4)], display_img4, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Axial - Slice: 06 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input4', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 

    image_plot([display_val5[i][-1] for i in range(4)], display_img5, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Sagittal - Slice: 58 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input5', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val6[i][-1] for i in range(4)], display_img6, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Coronal - Slice: 12 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))

    experiment_tag = '_'.join(['shap_image_input6', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 


def plot_bargraph(shap_values, plot_pred, predicted_class, true_class, plot_path, image_index = -1, fig_size = (8,8), palette = None):

    category_names = ['CN: ('+str(plot_pred[0].round(4))+')', 'sMCI: ('+str(plot_pred[1].round(4))+')', 'pMCI: ('+str(plot_pred[2].round(4))+')','AD: ('+str(plot_pred[3].round(4))+')']

    if predicted_class == true_class:
        choice = 'correctly'
    else:
        choice = 'wrongly'

    value_axial = np.transpose(shap_values[0][image_index], (3 ,0, 1, 2))
    value_sagittal = np.transpose(shap_values[1][image_index], (3 ,0, 1, 2))
    value_coronal = np.transpose(shap_values[2][image_index], (3 ,0, 1, 2))
    value_axial2 = np.transpose(shap_values[3][image_index], (3 ,0, 1, 2))
    value_sagittal2 = np.transpose(shap_values[4][image_index], (3 ,0, 1, 2))
    value_coronal2 = np.transpose(shap_values[5][image_index], (3 ,0, 1, 2))

    fig, ax = plt.subplots(figsize = (fig_size))
    sns.set_theme(palette)

    predicted_class_label = {0: 'a CN', 1: 'an sMCI', 2:'a pMCI', 3:'an AD'}

    labels = ('Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)')
    
    y_pos = np.arange(len(labels))

    sum_performance = [value_axial[0].sum().round(4),value_sagittal[0].sum().round(4),value_coronal[0].sum().round(4),
                value_axial2[0].sum().round(4),value_sagittal2[0].sum().round(4),value_coronal2[0].sum().round(4),
                value_axial[1].sum().round(4),value_sagittal[1].sum().round(4),value_coronal[1].sum().round(4),
                value_axial2[1].sum().round(4),value_sagittal2[1].sum().round(4),value_coronal2[1].sum().round(4),
                value_axial[2].sum().round(4),value_sagittal[2].sum().round(4),value_coronal[2].sum().round(4),
                value_axial2[2].sum().round(4),value_sagittal2[2].sum().round(4),value_coronal2[2].sum().round(4),
                value_axial[3].sum().round(4),value_sagittal[3].sum().round(4),value_coronal[3].sum().round(4),
                value_axial2[3].sum().round(4),value_sagittal2[3].sum().round(4),value_coronal2[3].sum().round(4)]

    ax.barh(y_pos[0:6], sum_performance[0:6], align='center', label = category_names[0])
    ax.barh(y_pos[6:12], sum_performance[6:12], align='center', label = category_names[1])
    ax.barh(y_pos[12:18], sum_performance[12:18], align='center', label = category_names[2])
    ax.barh(y_pos[18:24], sum_performance[18:24], align='center', label = category_names[3])

    ax.set_yticks(y_pos, labels=labels)
    ax.legend(ncols=1, fontsize='small', title = 'Predicted probabilities')
    ax.invert_yaxis()  
    ax.axvline(color="grey", alpha = 0.5)
    ax.set_xlabel('SHAP value')
    ax.set_ylabel('Orientation / Slice (Classifier)')
    ax.set_title('Sum of contributions of separate orientations to all model outputs\n for ' +str(predicted_class_label[true_class])+ ' patient ' + choice + ' classified as ' +str(predicted_class_label[predicted_class])+' patient' )

    experiment_tag = '_'.join(['shap_bar', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 


def plot_stripplot(shap_values, plot_pred, predicted_class, true_class, plot_path, num_images = 20, fig_size = (8,8), palette = None):

    fig, ax = plt.subplots(figsize = fig_size)
    sns.set_theme()

    if predicted_class == true_class:
        choice = 'correctly'
    else:
        choice = 'wrongly'

    labels = ('Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)')
    
    y_pos = np.arange(len(labels))

    predicted_class_label = {0: 'CN', 1: 'sMCI', 2:'pMCI', 3:'AD'}

    y_pos_jitter = [None]*24
    y_label_jitter = ['CN: ('+str(plot_pred[0].round(4))+')', 'sMCI: ('+str(plot_pred[1].round(4))+')', 'pMCI: ('+str(plot_pred[2].round(4))+')','AD: ('+str(plot_pred[3].round(4))+')']
    y_label_jitter = np.repeat(y_label_jitter, 6*num_images)

    for i in range(0, 24):

        y_pos_jitter[i] = np.repeat(y_pos[i], num_images)

    jitter_value_axial = np.transpose(shap_values[0], (0, 4 ,1, 2, 3))
    jitter_value_sagittal = np.transpose(shap_values[1], (0, 4 ,1, 2, 3))
    jitter_value_coronal = np.transpose(shap_values[2], (0, 4 ,1, 2, 3))
    jitter_value_axial2 = np.transpose(shap_values[3], (0, 4 ,1, 2, 3))
    jitter_value_sagittal2 = np.transpose(shap_values[4], (0, 4 ,1, 2, 3))
    jitter_value_coronal2 = np.transpose(shap_values[5], (0, 4 ,1, 2, 3))


    jitter_performance = [jitter_value_axial[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1)]

    data=pd.DataFrame({'col1': np.array(jitter_performance).flatten(), 'col2': np.array(y_pos_jitter).flatten(), 'col3' : y_label_jitter})

    sns.stripplot(x='col1', y='col2', data=data, hue = 'col3', palette=sns.color_palette(palette), orient = 'h')
    ax.set_yticks(y_pos, labels=labels)
    ax.legend(ncols=1, fontsize='small', title = 'Mean predicted probabilities')
    ax.axvline(color="grey",  alpha = 0.5)
    ax.grid(visible = True)
    ax.set_xlabel('SHAP value')
    ax.set_ylabel('Orientation / Slice (Classifier)')
    ax.set_title('Sums of input contributions to all model outputs\n for ' + str(num_images) + ' ' +str(predicted_class_label[true_class])+ ' patients ' + choice + ' classified as ' +str(predicted_class_label[predicted_class])+' patients')

    experiment_tag = '_'.join(['shap_strip', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 


def plot_extraplots(shap_values, predicted_class, true_class, plot_path, num_images = 20, fig_size = (8,18)):


    fig, axs = plt.subplots(nrows=2, ncols=1, figsize = fig_size)

    np.random.seed(19680801)

    predicted_class_label = {0: 'CN', 1: 'sMCI', 2:'pMCI', 3:'AD'}

    if predicted_class == true_class:
        choice = 'correctly'
    else:
        choice = 'wrongly'

    labels = ('Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)',
            'Axial / 12 (CN/MCI/AD)', 'Sagittal / 72 (CN/MCI/AD)', 'Coronal / 43 (CN/MCI/AD)','Axial / 06  (sMCI/pMCI)', 'Sagittal / 58  (sMCI/pMCI)', 'Coronal / 43  (sMCI/pMCI)')

    jitter_value_axial = np.transpose(shap_values[0], (0, 4 ,1, 2, 3))
    jitter_value_sagittal = np.transpose(shap_values[1], (0, 4 ,1, 2, 3))
    jitter_value_coronal = np.transpose(shap_values[2], (0, 4 ,1, 2, 3))
    jitter_value_axial2 = np.transpose(shap_values[3], (0, 4 ,1, 2, 3))
    jitter_value_sagittal2 = np.transpose(shap_values[4], (0, 4 ,1, 2, 3))
    jitter_value_coronal2 = np.transpose(shap_values[5], (0, 4 ,1, 2, 3))


    jitter_performance = [jitter_value_axial[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,0].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,1].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,2].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1), 
                        jitter_value_sagittal[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_axial2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_sagittal2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1),
                        jitter_value_coronal2[:,3].sum(axis = 1).sum(axis = 1).sum(axis = 1)]

    all_data =  jitter_performance


    # plot violin plot
    axs[0].axvline(color="grey", alpha = 0.5)
    axs[0].violinplot(all_data, showmeans=False, showmedians=True, vert = False)
    axs[0].set_title('Sums of input contributions to all model outputs\n for ' + str(num_images) + ' ' +str(predicted_class_label[true_class])+ ' patients ' + choice + ' classified as ' +str(predicted_class_label[predicted_class])+' patients')

    # plot box plot
    axs[1].axvline(color="grey", alpha = 0.5)
    axs[1].boxplot(all_data, vert = False)
    axs[1].set_title('Sums of input contributions to all model outputs\n for ' + str(num_images) + ' ' +str(predicted_class_label[true_class])+ ' patients ' + choice + ' classified as ' +str(predicted_class_label[predicted_class])+' patients')

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_yticks([y + 1 for y in range(len(all_data))],
                    labels=labels)
        ax.set_xlabel('SHAP values')
        ax.set_ylabel('Orientation / Slice (Classifier)')
        ax.invert_yaxis()

    experiment_tag = '_'.join(['shap_extraplots', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 

run(label_wanted = 0, label_true = 0, num_images = 25, num_samples = 500, plot_path = '/Users/olath/Documents')

