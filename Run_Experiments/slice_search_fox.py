#from dotenv import load_dotenv
import sys

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from experiment_manager import ExperimentManager, ExperimentParameters

from torchvision.transforms import RandAugment
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

from mri_dataset import MRI_Dataset

from copy import deepcopy

#load_dotenv()

import neptune
"""
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmQ2ZjBlYy04ZDQ0LTQ0ZjAtYWNhMS1hNzZlOWE0MTRmZDEifQ=="
NEPTUNE_PROJECT = "olathors-thesis/slice-test"
EXPERIMENT_PATH = "/experiments"
"""

np.random.seed(110323)
torch.manual_seed(110323)
#sys.path.append(os.path.abspath(os.path.join('..', 'src')))

#TODO Change to cuda only
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

#Parameters that do not change

# Running EfficientNet
optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss
batch_size = 32
learning_rate = 0.0001
epochs = 100
early_stopping_epochs = 20
experiment_tag = "Test slice search"
dataset_tag = "AD/MCI/CD"
model_tag="EfficientnetV2S"
num_classes = 3
weights_imagenet = None
transform_magnitude = 1
transform_num_ops = 2

job_id = sys.argv[0]

path = (job_id[0:22])

#Optimal model

model = efficientnet_v2_s(weights = weights_imagenet, num_classes = num_classes)

transform = RandAugment(magnitude = transform_magnitude, num_ops= transform_num_ops)

#train_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/train','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/', transform)
#val_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/val','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/')

orientation = 'coronal'

modelname = 's'

model_params = None

for slice in range(0, 101, 2):

    train_dataset = MRI_Dataset((path+'train'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED/'), transform, slice = slice)
    val_dataset = MRI_Dataset((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED/'), slice = slice)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    experiment_tag = ("Model: " + str(modelname) + ". Orientation: " + str(orientation) +  ". Slice index: " + str(slice))

    experiment_params = ExperimentParameters(
        model = deepcopy(model),
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
        transform_magnitude=transform_magnitude,
        transform_num_ops = transform_num_ops,
        experiment_directory = os.getenv("EXPERIMENT_PATH"),
    )

    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
    experiment_manager.run_experiment(train_loader, 
                                        validation_loader,
                                        save_logs=False,
                                        plot_results=False,
                                        verbose=0)

