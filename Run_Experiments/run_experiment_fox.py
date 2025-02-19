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
from customloss import FocalLoss

#load_dotenv()

import neptune

NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmQ2ZjBlYy04ZDQ0LTQ0ZjAtYWNhMS1hNzZlOWE0MTRmZDEifQ=="
NEPTUNE_PROJECT = "olathors-thesis/grid-test-4class"
EXPERIMENT_PATH = "/experiments"


np.random.seed(110323)
torch.manual_seed(110323)
#sys.path.append(os.path.abspath(os.path.join('..', 'src')))

def main():

    torch.multiprocessing.set_start_method('spawn')

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
    criterion = FocalLoss
    batch_size = 128
    learning_rate = 1e-4
    epochs = 100
    early_stopping_epochs = 20
    experiment_tag = "Test grid search"
    dataset_tag = "CN/sMCI/pMCI/AD"
    model_tag="EfficientnetV2S/M/L"
    num_classes = 4
    weights_imagenet = None
    transform_magnitude = 4
    weight = torch.Tensor([0.25, 0.25, 0.5])
    

    #Parameter grid
    model_s = efficientnet_v2_s(weights = None, num_classes = num_classes)
    #model_s = efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights)
    #model_s.classifier= torch.nn.Sequential(
    #                torch.nn.Dropout(p=0.2, inplace=True),
    #                torch.nn.Linear(1280, num_classes),
    #            )
    model_m = efficientnet_v2_m(weights = None, num_classes = num_classes) 
    #model_m = efficientnet_v2_m(weights = torchvision.models.EfficientNet_V2_M_Weights)
    #model_m.classifier= torch.nn.Sequential(
    #                torch.nn.Dropout(p=0.3, inplace=True),
    #                torch.nn.Linear(1280, num_classes),
    #            )
    model_l = efficientnet_v2_l(weights = None, num_classes = num_classes)
    #model_l  = efficientnet_v2_l(weights = torchvision.models.EfficientNet_V2_L_Weights)
    #model_l.classifier= torch.nn.Sequential(
    #                torch.nn.Dropout(p=0.4, inplace=True),
    #                torch.nn.Linear(1280, num_classes),
    #            )

    param_grid = {
        'model' : [model_s, model_m, model_l],
        'batch_size' : [16],
        'learning_rate' : [0.00001, 0.000001],
        'optimizer' :[torch.optim.Adam],
        'loss' : [FocalLoss],
        'epochs' : [200],
        'early_stopping_epochs' : [25],
        'transform_magnitude' : [1, 2],
        'transform_num_ops' : [1, 2],
        'triple' : [False],
        'alpha' : [[0.25,0.25,0.5]],
        'gamma' : [1, 2, 3, 4, 5]
    }

    #train_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/train','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/', transform)
    #val_dataset = MRI_Dataset('/fp/homes01/u01/ec-olathor/Documents/thesis/val','/fp/homes01/u01/ec-olathor/Documents/thesis/ADNI_SLICED/')

    job_id = sys.argv[0]

    path = (job_id[0:22])

    modelname = 0

    model_params = None
    for model in param_grid['model']:
        modelname += 1
        for batch_size in param_grid['batch_size']:
            for learning_rate in param_grid['learning_rate']:
                for optimizer in param_grid['optimizer']:
                    for criterion in param_grid['loss']:
                        for epochs in param_grid['epochs']:
                            for early_stopping_epochs in param_grid['early_stopping_epochs']:
                                for transform_magnitude in param_grid['transform_magnitude']:
                                    for transform_num_ops in param_grid['transform_num_ops']:
                                        for triple in param_grid['triple']:
                                            for alpha in param_grid['alpha']:
                                                for gamma in param_grid['gamma']:


                                                    #Checking for redundant experiment runs with transformation magnitude zero
                                                    if transform_magnitude == 0 and transform_num_ops == 1:
                                                        #This is the only case where no transform happens
                                                        transform = None
                                                        transform_num_ops = 0
                                                    elif transform_magnitude == 0:
                                                        #In this case, we skip the experiment.
                                                        break                
                                                    else:
                                                        transform = RandAugment(magnitude = transform_magnitude, num_ops= transform_num_ops)

                                                    train_dataset = MRI_Dataset((path+'train'), (path+'ADNI_SLICED/'), transform = transform, triple = triple)
                                                    val_dataset = MRI_Dataset((path+'val'), (path+'ADNI_SLICED/'), triple = triple)

                                                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                                    validation_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                                                    experiment_tag = ("Weighted focal loss by ratio. Model: " + str(modelname) + ".Batch size: " + str(batch_size) + ".Learning_rate: " + str(learning_rate) + '.Transform magnitude: ' + 
                                                                    str(transform_magnitude) + '.Transform num ops: ' + str(transform_num_ops) + '.  ')

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
                                                        NEPTUNE_PROJECT = NEPTUNE_PROJECT,
                                                        experiment_directory = os.getenv("EXPERIMENT_PATH"),
                                                        classes = num_classes,
                                                        alpha = alpha,
                                                        gamma = gamma,
                                                        weight = weight
                                                    )

                                                    experiment_manager = ExperimentManager(experiment_parameters=experiment_params)
                                                    experiment_manager.run_experiment(train_loader, 
                                                                                        validation_loader,
                                                                                        save_logs=False,
                                                                                        plot_results=False,
                                                                                        verbose=0)
                                                
if __name__ == '__main__':
    main()

