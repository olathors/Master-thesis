#from dotenv import load_dotenv


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
NEPTUNE_PROJECT = "olathors-thesis/grid-test"
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
batch_size = 128
learning_rate = 1e-4
epochs = 100
early_stopping_epochs = 20
experiment_tag = "Test experiment AD/MCI/CD"
dataset_tag = "AD/MCI/CD"
model_tag="EfficientnetV2m imagenet"
num_classes = 3
weights_imagenet = None
transform_magnitude = 4

#Parameter grid

param_grid = {
    'batch_size' : [8,16,32,64],
    'learning_rate' : [0.001, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001],
    'optimizer' :[torch.optim.Adam, torch.optim.SGD],
    'loss' : [torch.nn.CrossEntropyLoss],
    'epochs' : [100],
    'early_stopping_epochs' : [50],
    'transform_magnitude' : [0, 1, 2, 3, 4],
    'transform_num_ops' : [1, 2, 3, 4, 5]
}

#train_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/train','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/', transform)
#val_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/val','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/')

model = efficientnet_v2_s(weights = weights_imagenet, num_classes = num_classes)


model_params = None

for batch_size in param_grid['batch_size']:
    for learning_rate in param_grid['learning_rate']:
        for optimizer in param_grid['optimizer']:
            for criterion in param_grid['loss']:
                for epochs in param_grid['epochs']:
                    for early_stopping_epochs in param_grid['early_stopping_epochs']:
                        for transform_magnitude in param_grid['transform_magnitude']:
                            for transform_num_ops in param_grid['transform_num_ops']:

                                if transform_magnitude == 0:
                                    transform = None
                                else:
                                    transform = RandAugment(magnitude = transform_magnitude, num_ops= transform_num_ops)

                                train_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/train', '/localscratch/1325070/ADNI_SLICED/', transform)
                                val_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/val', '/localscratch/1325070/ADNI_SLICED/')

                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                                experiment_tag = ("Batch size: " + str(batch_size) + ". Learning_rate: " + str(learning_rate) +  
                                                ". Optimizer: " + str(optimizer) + ". Loss: " + str(criterion) + ". Epochs: " + 
                                                str(epochs) + " . Early stopping: " + str(early_stopping_epochs) + '. Transform magnitude: ' + 
                                                str(transform_magnitude) + '. Transform num ops: ' + str(transform_num_ops) + '. ----')

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
                                                                    verbose=1)

