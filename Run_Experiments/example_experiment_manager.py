import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

import torch
from torch.nn import CrossEntropyLoss
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from transformers import ViTForImageClassification
#import timm

from tqdm import tqdm, trange
from dotenv import load_dotenv
from loguru import logger

from mri_dataset import MRI_Dataset
from torch.utils.data import DataLoader

#from custom_efficientnet import CustomEfficientnet
from torchvision.models.efficientnet import _efficientnet_conf, efficientnet_v2_l, efficientnet_v2_s, EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
from torchvision.transforms import functional, RandAugment

from customloss import FocalLoss
from combined_classifier import CombinedClassifierL
#import custom_efficientnet

#%load_ext autoreload
#%autoreload 2

load_dotenv()

np.random.seed(0)
torch.manual_seed(0)
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from experiment_manager import *

def main():

    # setting device on GPU if available, else CPU
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
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
    epochs = 1
    early_stopping_epochs = 25
    experiment_tag = "4class"
    dataset_tag = "AD/sMCI/pMCI/CD"
    model_tag="EfficientnetV2L"
    num_classes = 2
    weights_imagenet = EfficientNet_V2_S_Weights.DEFAULT
    transform_magnitude = 14
    transform_num_ops = 4
    alpha = 0
    gamma = 0
    weight = None


    transform = RandAugment(magnitude = transform_magnitude, num_ops = transform_num_ops)

    train_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/Datasets/train-sMCI-pMCI','/Users/olath/Documents/ADNI_SLICED_RESCALED/', transform=transform, slice= 6, orientation= 'AXIAL') 
    val_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/Datasets/val-sMCI-pMCI','/Users/olath/Documents/ADNI_SLICED_RESCALED/', slice= 6, orientation= 'AXIAL')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    """
    axial = efficientnet_v2_l(num_classes = 4)
    axial.load_state_dict(torch.load('/Users/olath/Downloads/model_12AXIAL_202502281555_best.pth', weights_only=True, map_location=torch.device('mps')))
    sagittal = efficientnet_v2_l(num_classes = 4)
    sagittal.load_state_dict(torch.load('/Users/olath/Downloads/model_72SAGITTAL_202502281601_best.pth', weights_only=True, map_location=torch.device('mps')))
    coronal = efficientnet_v2_l(num_classes = 4)
    coronal.load_state_dict(torch.load('/Users/olath/Downloads/model_43CORONAL_202502281527_best.pth', weights_only=True, map_location=torch.device('mps')))
    

    model = CombinedClassifier(4, axial, sagittal, coronal)
    """

    model  = efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
    #nn.BatchNorm1d(num_features=1280),    
    nn.Linear(1280, 2),
    #nn.ReLU(),
    #nn.BatchNorm1d(512),
    #nn.Linear(512, 128),
    #nn.ReLU(),
    #nn.BatchNorm1d(num_features=128),
    #nn.Dropout(0.4),
    #nn.Linear(128, 2),
    )


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
        class_id = {0:'sMCI',1:'pMCI'} 
    )

    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
    experiment_manager.run_experiment(train_loader, 
                                    validation_loader,
                                    save_logs=False,
                                    plot_results=False,
                                    verbose=1)
    
    
if __name__ == '__main__':
    main()