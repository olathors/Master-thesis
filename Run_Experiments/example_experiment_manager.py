import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.nn import CrossEntropyLoss
import torchvision
#from transformers import ViTForImageClassification
#import timm

from tqdm import tqdm, trange
from dotenv import load_dotenv
from loguru import logger

from mri_dataset import MRI_Dataset
from torch.utils.data import DataLoader

#%load_ext autoreload
#%autoreload 2

load_dotenv()

np.random.seed(0)
torch.manual_seed(0)
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from experiment_manager import *
#from dataset import load_cifar10,load_cifar100
#from model import load_model

# setting device on GPU if available, else CPU
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')



# Running ViT Tiny
optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss
batch_size = 16
learning_rate = 1e-4
epochs = 100
early_stopping_epochs = 10
experiment_tag = "Test experiment"
dataset_tag = "Custom dataset"
num_classes = 2

train_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/train')
test_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/test')
val_dataset = MRI_Dataset('/Users/olath/Documents/GitHub/Master-thesis/val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

#train_loader, validation_loader = load_cifar100(batch_size=batch_size,validation_split=0.5)
model = torchvision.models.efficientnet_v2_l(num_classes = 2)
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
    model_tag="ViT-tiny",
    experiment_directory = os.getenv("EXPERIMENT_PATH"),
)

experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
experiment_manager.run_experiment(train_loader, 
                                  validation_loader,
                                  save_logs=False,
                                  plot_results=True,
                                  verbose=1)