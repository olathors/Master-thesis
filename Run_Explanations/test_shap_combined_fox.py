import sys
import os
sys.path.append('../')
import shap
import torch
from torch.utils.data import DataLoader
from mri_dataset_combined_shap import MRI_Dataset_combined
from torchvision.models.efficientnet import efficientnet_v2_m
import numpy as np
from combined_classifier_shap import *
import matplotlib.pyplot as plt
from custom_image_shap import image as image_plot
import seaborn as sns
import pandas as pd
from datetime import datetime
import time


def run(label_wanted, label_true, num_images, num_samples, image_index, plot_path, even_spread = False):

    seed = 110323
    np.random.seed(seed)
    torch.manual_seed(seed)

    sns.set_theme(palette = "deep")

    start = time.time()
    print('Starting experiment')

    just_labels = {0: 'CN', 1:'sMCI',2:'pMCI',3:'AD'}

    experiment_name = 'experiment_predlabel-'+str(just_labels[label_wanted])+'_truelabel-'+str(just_labels[label_true])+'_numimages-'+str(num_images)+'_numsamples-'+str(num_samples)+'_time-'+datetime.now().strftime("%Y%m%d%H%M")

    print('Setting up models at %.2f minutes' % ((time.time() - start)/60))
    print(experiment_name)

    plot_path = os.path.join(plot_path, experiment_name)

    if not os.path.exists(plot_path ):
        os.makedirs(plot_path)

    plt.style.use('seaborn-v0_8') 

    device = torch.device('cuda')

    job_id = sys.argv[0]

    path = (job_id[0:22])

    #Initializing datasets
    train_dataset = MRI_Dataset_combined((path+'train'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), device, slice= [12, 72, 43, 6, 58, 43])
    val_dataset = MRI_Dataset_combined((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), device, slice= [12, 72, 43, 6, 58, 43])

    train_loader  = DataLoader(train_dataset, batch_size=500, shuffle=True)
    validation_loader  = DataLoader(val_dataset, batch_size=1070, shuffle=True)

    #Initializing models
    axial1 = efficientnet_v2_m(num_classes = 3)
    sagittal1 = efficientnet_v2_m(num_classes = 3)
    coronal1 = efficientnet_v2_m(num_classes = 3)
    axial2 = efficientnet_v2_m(num_classes = 2)
    sagittal2 = efficientnet_v2_m(num_classes = 2)
    coronal2 = efficientnet_v2_m(num_classes = 2)

    model = DoubleCombinedClassifierLogReg(4, axial1, sagittal1, coronal1, axial2, sagittal2, coronal2, device, dropout = 0.4, num_outputs = 250)
    model.load_state_dict(torch.load('/fp/projects01/ec29/olathor/thesis/saved_models/model_4classDouble_202503171605_best.pth', weights_only=True, map_location=torch.device('cuda')))
    model.to(device)
    model.eval()

    print('Finding images at %.2f minutes' % ((time.time() - start)/60))

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

    if len(scores) < num_images:
        print('Not enough images, only found ' + str(len(scores)) + ' out of the required ' + str(num_images))
        sys.exit()
    else:
        print('Found ' + str(len(scores)) + ' images')
        
    for score in scores:
        best.append([X[0][best_dict[score]].cpu().detach(), X[1][best_dict[score]].cpu().detach(), X[2][best_dict[score]].cpu().detach(), 
                X[3][best_dict[score]].cpu().detach(), X[4][best_dict[score]].cpu().detach(), X[5][best_dict[score]].cpu().detach()])

    best_pred = best_pred.cpu().detach().numpy().round(4)

    y_num = y[best_pred_index:best_pred_index+1].numpy()

    labels = {0: 'True class: CN\nPredicted: ' +str(just_labels[label_wanted]), 1:'True class: sMCI\nPredicted: ' +str(just_labels[label_wanted]), 2: 'True class: pMCI\nPredicted: ' +str(just_labels[label_wanted]), 3:'True class: AD\nPredicted: ' +str(just_labels[label_wanted])}
    true_labels = list()

    for  label in y_num:    

        true_labels.append(labels[label])

    #Setting up and running the explainer
    X_train, y_train = next(iter(train_loader))

    best = np.array(best)

    #Picking n evenly spaced images in the sorted list of lowest to highest prediction scores.
    if even_spread:
        idx = np.round(np.linspace(0, len(best) - 1, num_images)).astype(int)
        best_input = np.transpose(best[idx], (1, 0, 2, 3, 4)).tolist()
    else:
        best_input = np.transpose(best[-num_images:], (1, 0, 2, 3, 4)).tolist()

    

    for i in range(0, len(best_input)):
        best_input[i] = torch.tensor(best_input[i]).to(device)

    print('Setting up explainer at %.2f minutes' % ((time.time() - start)/60))

    explainer = shap.GradientExplainer(model, X_train, batch_size=16)

    print('Running explainer at %.2f minutes' % ((time.time() - start)/60))

    shap_values = explainer.shap_values(best_input, nsamples = num_samples, rseed = seed)

    print('Creating plots at %.2f minutes' % ((time.time() - start)/60))

    pred = model(best_input[0],best_input[1],best_input[2],best_input[3],best_input[4],best_input[5]).detach().cpu().numpy()

    pred_means = [pred[:,0].mean(),pred[:,1].mean(),pred[:,2].mean(),pred[:,3].mean()]

    for index in image_index:

        plot_shap_images(X, shap_values, true_labels, pred, plot_path, index)

        plot_bargraph(shap_values, pred, label_wanted, label_true, plot_path, image_index = index, fig_size = (8,8), palette = "deep")

    plot_stripplot(shap_values, pred_means, label_wanted, label_true, plot_path, num_images = num_images, fig_size = (8,8), palette = "deep")

    plot_extraplots(shap_values, label_wanted, label_true, plot_path, num_images = num_images, fig_size = (8,18))

    print('Finished at %.2f minutes' % ((time.time() - start)/60))


def plot_shap_images(X, shap_values, true_labels,  pred, plot_path, image_index = -1):

    plot_pred = [pred[image_index,0].round(4),pred[image_index,1].round(4),pred[image_index,2].round(4),pred[image_index,3].round(4)]

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
    
    image_plot([display_val1[i][image_index] for i in range(4)], display_img1, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Axial - Slice: 12 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))

    experiment_tag = '_'.join(['shap_image_input1_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val2[i][image_index] for i in range(4)], display_img2, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Sagittal - Slice: 72 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input2_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val3[i][image_index] for i in range(4)], display_img3, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Coronal - Slice: 43 - Classifier: CN/MCI/AD - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input3_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val4[i][image_index] for i in range(4)], display_img4, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Axial - Slice: 06 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input4_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 

    image_plot([display_val5[i][image_index] for i in range(4)], display_img5, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Sagittal - Slice: 58 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))
    
    experiment_tag = '_'.join(['shap_image_input5_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 
    
    image_plot([display_val6[i][image_index] for i in range(4)], display_img6, show = False, labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'], true_labels = true_labels, title = "Orientation: Coronal - Slice: 12 - Classifier: sMCI/pMCI - Predicted probabilities: " + str(plot_pred))

    experiment_tag = '_'.join(['shap_image_input6_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 


def plot_bargraph(shap_values, pred, predicted_class, true_class, plot_path, image_index = -1, fig_size = (8,8), palette = None):

    plot_pred = [pred[image_index,0].round(4),pred[image_index,1].round(4),pred[image_index,2].round(4),pred[image_index,3].round(4)]

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

    ax.barh(y_pos[0:6], sum_performance[0:6], align='center', label = category_names[0], color = sns.color_palette(palette).as_hex()[0])
    ax.barh(y_pos[6:12], sum_performance[6:12], align='center', label = category_names[1], color = sns.color_palette(palette).as_hex()[1])
    ax.barh(y_pos[12:18], sum_performance[12:18], align='center', label = category_names[2], color = sns.color_palette(palette).as_hex()[2])
    ax.barh(y_pos[18:24], sum_performance[18:24], align='center', label = category_names[3], color = sns.color_palette(palette).as_hex()[3])

    ax.set_yticks(y_pos, labels=labels)
    ax.legend(ncols=1, fontsize='small', title = 'Predicted probabilities')
    ax.invert_yaxis()  
    ax.axvline(color="grey", alpha = 0.5)
    ax.set_xlabel('SHAP value')
    ax.set_ylabel('Orientation / Slice (Classifier)')
    ax.set_title('Sum of contributions of separate orientations to all model outputs\n for ' +str(predicted_class_label[true_class])+ ' patient ' + choice + ' classified as ' +str(predicted_class_label[predicted_class])+' patient' )

    experiment_tag = '_'.join(['shap_bar_index('+str(image_index)+')', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plot_path, f'plot_{experiment_tag}.pdf')

    plt.savefig(plot_save_path, dpi = 1200, bbox_inches='tight')
    plt.close() 


def plot_stripplot(shap_values, plot_pred, predicted_class, true_class, plot_path, num_images = 20, fig_size = (8,8), palette = None):

    fig, ax = plt.subplots(figsize = fig_size)

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

    sns.stripplot(x='col1', y='col2', data=data, hue = 'col3', palette=sns.color_palette(palette)[:4], orient = 'h')
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

run(label_wanted = 0, label_true = 0, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 1, label_true = 1, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 2, label_true = 2, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 3, label_true = 3, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')

run(label_wanted = 0, label_true = 1, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 1, label_true = 2, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 2, label_true = 3, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')

run(label_wanted = 0, label_true = 2, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')
run(label_wanted = 1, label_true = 3, num_images = 20, num_samples = 1000, image_index = [19, 10, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')

run(label_wanted = 0, label_true = 3, num_images = 15, num_samples = 1000, image_index = [14, 7, 0], plot_path = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots')

