import os
import sys

import numpy as np

import torch

from mri_dataset_combined import MRI_Dataset
from torch.utils.data import DataLoader

from torchvision.models.efficientnet import _efficientnet_conf, efficientnet_v2_l, efficientnet_v2_s, EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
from torchvision.transforms import functional, RandAugment

from combined_classifier import CombinedClassifier

np.random.seed(0)
torch.manual_seed(0)
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from experiment_manager import *

def main():

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    # Running EfficientNet
    optimizer = torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss
    batch_size = 16
    learning_rate = 0.00001
    epochs = 200
    early_stopping_epochs = 25
    experiment_tag = "4class"
    dataset_tag = "AD/sMCI/pMCI/CD"
    model_tag="EfficientnetV2COmbined"
    num_classes = 4
    #weights_imagenet = EfficientNet_V2_S_Weights.DEFAULT
    transform_magnitude = 14
    transform_num_ops = 4
    alpha = 0
    gamma = 0
    weight = None

    job_id = sys.argv[0]

    path = (job_id[0:22])

    transform = RandAugment(magnitude = transform_magnitude, num_ops = transform_num_ops)

    train_dataset = MRI_Dataset((path+'train'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), transform=transform, slice= [12, 72, 43]) 
    val_dataset = MRI_Dataset((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), slice= [12, 72, 43])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    axial = efficientnet_v2_l(num_classes = 4)
    axial.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_12AXIAL_202503010344_best.pth', weights_only=True, map_location=torch.device('cuda')))
    sagittal = efficientnet_v2_l(num_classes = 4)
    sagittal.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_72SAGITTAL_202502281601_best.pth', weights_only=True, map_location=torch.device('cuda')))
    coronal = efficientnet_v2_l(num_classes = 4)
    coronal.load_state_dict(torch.load('/fp/homes01/u01/ec-olathor/Documents/thesis/model_44CORONAL_202502280117_best.pth', weights_only=True, map_location=torch.device('cuda')))
    

    model = CombinedClassifier(4, axial, sagittal, coronal, tune_classifiers= True, dropout= 0.1)
    model_params = None

    experiment_params = ExperimentParameters(
        model=model,
        model_params=model_params,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_epochs=early_stopping_epochs,
        experiment_tag=experiment_tag,
        dataset_tag=dataset_tag,
        model_tag=model_tag,
        experiment_directory = os.getenv("EXPERIMENT_PATH"),
        classes = num_classes,
        alpha= alpha,
        gamma = gamma,
        weight = weight,
        class_id = {0:'CN',1:'sMCI',2:'pMCI', 3:'AD'} 
    )

    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
    experiment_manager.run_experiment(train_loader, 
                                    validation_loader,
                                    save_logs=False,
                                    plot_results=False,
                                    verbose=1)
    
    
if __name__ == '__main__':
    main()