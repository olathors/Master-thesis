#from dotenv import load_dotenv
import sys

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from experiment_manager import ExperimentManager, ExperimentParameters

from torchvision.transforms import RandAugment
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
import torchvision

from mri_dataset import MRI_Dataset

from copy import deepcopy

#load_dotenv()

import neptune

NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmQ2ZjBlYy04ZDQ0LTQ0ZjAtYWNhMS1hNzZlOWE0MTRmZDEifQ=="
NEPTUNE_PROJECT = "olathors-thesis/test"
EXPERIMENT_PATH = "/fp/projects01/ec29/olathor/thesis/saved_models"

def main():
    #torch.multiprocessing.set_start_method('spawn', force = True)

    np.random.seed(110323)
    torch.manual_seed(110323)
    #sys.path.append(os.path.abspath(os.path.join('..', 'src')))

    #TODO Change to cuda only
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    batch_size = 16
    learning_rate = [0.00001, 0.000005]
    epochs = 200
    early_stopping_epochs = 25
    experiment_tag = "Slice search extended"
    dataset_tag = "CN/sMCI/pMCI/AD"
    model_tag="EfficientnetV2L"
    num_classes = 4
    #weights_imagenet = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    transform_magnitude = [12, 14, 16]
    transform_num_ops = [2, 3, 4]
    dropout = [0.3, 0.4, 0.5]

    job_id = sys.argv[0]

    path = (job_id[0:22])

    #Optimal model

    model  = efficientnet_v2_l(weights = torchvision.models.EfficientNet_V2_L_Weights.DEFAULT)
 
    slice_dict = {'SAGITTAL':73,
                  'AXIAL':13,
                  'CORONAL':41}

    #train_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/train','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/', transform)
    #val_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/val','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/')

    orientation = 'CORONAL'

    modelname = 'L'

    model_params = None
    for learning_rate in learning_rate:
        for transform_magnitude in transform_magnitude:
            for transform_num_ops in transform_num_ops:
                for dropout in dropout:

                    model.classifier= torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout, inplace=True),
                    torch.nn.Linear(1280, num_classes),)

                    slice = slice_dict[orientation]

                    transform = RandAugment(magnitude = transform_magnitude, num_ops= transform_num_ops)
                
                    train_dataset = MRI_Dataset((path+'train'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), transform = transform, slice = slice, orientation = orientation)
                    val_dataset = MRI_Dataset((path+'val'), ('/fp/projects01/ec29/olathor/thesis/ADNI_SLICED_RESCALED/'), slice = slice, orientation = orientation)

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


                    NEPTUNE_PROJECT = ("olathors-thesis/grid-test-4class-pretrained-final") 

                    experiment_tag = ('Orientation: ' + orientation + ' Magnitude: ' + str(transform_magnitude) + ' Ops: ' + str(transform_num_ops)+ ' Dropout' + str(dropout))

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
                        EXPERIMENT_PATH = EXPERIMENT_PATH,
                        NEPTUNE_API_TOKEN = NEPTUNE_API_TOKEN,
                        NEPTUNE_PROJECT = NEPTUNE_PROJECT,
                        classes = num_classes,
                        class_id = {0:'CN',1:'MCI', 2:'AD'}
                    )

                    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
                    experiment_manager.run_experiment(train_loader, 
                                                        validation_loader,
                                                        save_logs=False,
                                                        plot_results=False,
                                                        verbose=0,
                                                        save_best_model= True)


if __name__ == '__main__':
    main()

