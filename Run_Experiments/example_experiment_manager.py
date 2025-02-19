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

from custom_efficientnet import CustomEfficientnet
from torchvision.models.efficientnet import _efficientnet_conf
from torchvision.transforms import functional, RandAugment

from customloss import FocalLoss

#%load_ext autoreload
#%autoreload 2

load_dotenv()

np.random.seed(0)
torch.manual_seed(0)
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from experiment_manager import *

def main():

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    criterion = FocalLoss
    batch_size = 16
    learning_rate = 0.00001
    epochs = 200
    early_stopping_epochs = 50
    experiment_tag = "2class"
    dataset_tag = "AD/MCI/CD"
    model_tag="EfficientnetV2S"
    num_classes = 4
    weights_imagenet = None
    transform_magnitude = 2
    transform_num_ops = 1
    alpha = 0.25
    gamma = 2
    weight = torch.Tensor([0.25,0.25,0.5,0.5])


    transform = RandAugment(magnitude = transform_magnitude, num_ops = transform_num_ops)

    train_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/train_CN-sMCI-pMCI-AD','/Users/olath/Documents/ADNI_SLICED/', transform, triple = True) 
    #test_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/test')
    val_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/val_CN-sMCI-pMCI-AD','/Users/olath/Documents/ADNI_SLICED/', triple = True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = torchvision.models.efficientnet_v2_l(weights = weights_imagenet, num_classes = num_classes)

    if weights_imagenet is not None:

        #Dropout for s model = 0.2
        #Dropout for m model = 0.3, and linear layer in is 1280.
        #Dropout for l model = 0.4

        dropout = 0.2
        linear_in = 1280

        model.classifier= torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout, inplace=True),
                    torch.nn.Linear(linear_in, num_classes),
                )
        
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
        

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
        weight = weight
    )

    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
    experiment_manager.run_experiment(train_loader, 
                                    validation_loader,
                                    save_logs=False,
                                    plot_results=False,
                                    verbose=1)
    
    
if __name__ == '__main__':
    main()