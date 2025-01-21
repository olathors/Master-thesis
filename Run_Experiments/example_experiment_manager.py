import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.nn import CrossEntropyLoss
from transformers import ViTForImageClassification
import timm

from tqdm import tqdm, trange
from dotenv import load_dotenv
from loguru import logger

%load_ext autoreload
%autoreload 2

load_dotenv()

np.random.seed(0)
torch.manual_seed(0)
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from experiment_manager import *
from dataset import load_cifar10,load_cifar100
from model import load_model

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
batch_size = 128
learning_rate = 1e-3
epochs = 20
early_stopping_epochs = 5
experiment_tag = "ViT-tiny - CIFAR100 reduced"
dataset_tag = "CIFAR100-reduced"
num_classes = 100

train_loader, validation_loader = load_cifar100(batch_size=batch_size,validation_split=0.5)
model = load_model(model_type='vit-tiny', num_classes=num_classes)
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
                                  save_logs=True,
                                  plot_results=True,
                                  verbose=1)