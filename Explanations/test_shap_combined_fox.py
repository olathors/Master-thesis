import sys
import os
import shap
import torch
import cv2
from torch.utils.data import DataLoader
from mri_dataset_combined_shap import MRI_Dataset
from torchvision.models.efficientnet import efficientnet_v2_l
import numpy 
import torchvision
from combined_classifier_shap import CombinedClassifierL 
import matplotlib.pyplot as plt
from datetime import datetime
from custom_image_shap import image as image_plot

job_id = sys.argv[0]

path = (job_id[0:22])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
 
val_dataset = MRI_Dataset((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), slice= [12, 72, 43])

validation_loader  = DataLoader(val_dataset, batch_size=16, shuffle=True)

axial = efficientnet_v2_l(num_classes = 4)
axial.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_12AXIAL_202503010344_best.pth', weights_only=True, map_location=torch.device('cuda')))
sagittal = efficientnet_v2_l(num_classes = 4)
sagittal.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_72SAGITTAL_202502281601_best.pth', weights_only=True, map_location=torch.device('cuda')))
coronal = efficientnet_v2_l(num_classes = 4)
coronal.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_43CORONAL_202502281527_best.pth', weights_only=True, map_location=torch.device('cuda')))

model = CombinedClassifierL(4, axial, sagittal, coronal, dropout= 0.8)
model.load_state_dict(torch.load('/fp/projects01/ec29/olathor/thesis/saved_models/model_4class_202503031601_best.pth', weights_only=True, map_location=torch.device('cuda')))
model.to(device)
model.eval()

def f(x):
    x = torch.from_numpy(x)
    return model(x.to(device))

X, y = next(iter(validation_loader))

y_num = y[2:3].numpy()

labels = {0: 'True class: CN', 1:'True class: sMCI', 2: 'True class: pMCI', 3:'True class: AD'}
true_labels = list()

for  label in y_num:    

    true_labels.append(labels[label])


masker = shap.maskers.Image("blur(128,128)", X[0].shape)


explainer = shap.Explainer(f, masker)

shap_values = explainer(X[2:3], max_evals=1000000, batch_size=64, outputs=shap.Explanation.argsort.flip[:4])

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