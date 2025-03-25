import sys
import os
import shap
import torch
import cv2
from torch.utils.data import DataLoader
from mri_dataset_onemodel_shap import MRI_Dataset
from torchvision.models.efficientnet import efficientnet_v2_l
import numpy 
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
from custom_image_shap import image as image_plot

job_id = sys.argv[0]

path = (job_id[0:22])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
 

def explain( device, orientation, slice, label_wanted = 0, max_evals=5000000):

    val_dataset = MRI_Dataset((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), slice= slice, orientation = orientation)

    validation_loader  = DataLoader(val_dataset, batch_size=64, shuffle=True)

    if orientation == 'AXIAL':
        model = efficientnet_v2_l(num_classes = 4)
        model.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_12AXIAL_202503010344_best.pth', weights_only=True, map_location=torch.device('cuda')))
    elif orientation == 'SAGITTAL':
        model = efficientnet_v2_l(num_classes = 4)
        model.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_72SAGITTAL_202502281601_best.pth', weights_only=True, map_location=torch.device('cuda')))
    elif orientation == 'CORONAL':
        model = efficientnet_v2_l(num_classes = 4)
        model.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_43CORONAL_202502281527_best.pth', weights_only=True, map_location=torch.device('cuda')))
    else:
        print('Could not find orientation.')

    model.to(device)
    model.eval()

    X, y = next(iter(validation_loader))

    def f(x):
        x = torch.from_numpy(x)
        x = torch.permute(x,(0, 3, 1, 2))
        return torch.softmax(model(x.to(device)), 1)

    out = f(X.numpy()).cpu().detach().numpy()

    best_pred = 0
    best_pred_index = 0

    for j in range (0, 18):
        out = f(X.numpy()).cpu().detach().numpy()
        for i in range(0, len(out)):
            if y[i] == label_wanted:# and ((sum(out[i]) - out[i][label_wanted]) < 0.05):
                if out[i][label_wanted] > best_pred:
                    best_pred = out[i][label_wanted]
                    best_pred_index = i
        #out = f(X.numpy()).cpu().detach().numpy()

    print(best_pred)

    out = f(X[best_pred_index:best_pred_index+1].numpy()).cpu().detach().numpy()

    print(out)

    y_num = y[best_pred_index:best_pred_index+1].numpy()

    labels = {0: 'True class: CN', 1:'True class: sMCI', 2: 'True class: pMCI', 3:'True class: AD'}
    true_labels = list()

    for  label in y_num:    

        true_labels.append(labels[label])


    masker = shap.maskers.Image("blur(128,128)", X[0].shape)


    explainer = shap.Explainer(f, masker)

    shap_values = explainer(X[best_pred_index:best_pred_index+1], max_evals=max_evals, batch_size=64)

    shap_values_values = [val for val in numpy.moveaxis(shap_values.values, -1, 0)]

    data = shap_values.data

    data_num = data.numpy()
    data_scaled = data_num/data_num.max()

    image_plot(
        shap_values=shap_values_values,
        pixel_values=data_scaled,
        labels = ['Contributions to CN', 'Contributions to sMCI', 'Contributions to pMCI', 'Contributions to AD'],
        true_labels = true_labels,
        show = False
    )

    plotpath = '/fp/projects01/ec29/olathor/thesis/saved_shap_plots'

    experiment_tag = '_'.join(['SHAP_testplot', datetime.now().strftime("%Y%m%d%H%M")])
    plot_save_path = os.path.join(plotpath, f'model_{experiment_tag}.png')

    plt.savefig(plot_save_path, dpi = 1200) 

explain(device, 'AXIAL', 12, label_wanted = 0, max_evals=5000000)
explain(device, 'AXIAL', 12, label_wanted = 1, max_evals=5000000)
explain(device, 'AXIAL', 12, label_wanted = 2, max_evals=5000000)
explain(device, 'AXIAL', 12, label_wanted = 3, max_evals=5000000)