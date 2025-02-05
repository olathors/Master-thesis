import os 
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

import torch
from torch.nn import Module

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import neptune

NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
EXPERIMENT_PATH = os.getenv('EXPERIMENT_PATH','~/projects/phd/experiments')

device = torch.device('mps' if torch.mps.is_available() else 'cpu')

from dataclasses import dataclass

@dataclass
class ExperimentParameters:
    """
    A class to hold parameters for an experiment.

    Attributes:
    ----------
    model : Module
        The model to be used in the experiment.
    optimizer : Any
        The optimizer to be used for training the model.
    criterion : Any
        The loss function to be used during training.
    model_params : Optional[dict], default=None
        Additional parameters for the model.
    scheduler : Optional[Any], default=None
        The learning rate scheduler to be used during training.
    batch_size : Optional[int], default=512
        The number of samples per batch.
    learning_rate : Optional[float], default=0.001
        The learning rate for the optimizer.
    epochs : Optional[int], default=100
        The number of epochs to train the model.
    early_stopping_epochs : Optional[int], default=10
        The number of epochs with no improvement after which training will be stopped.
    
    pruning : Optional[bool], default=False
        Whether to use pruning during training.
    experiment_tag : Optional[str], default='SWViT_CIFAR10'
        A tag to identify the experiment.
    dataset_tag : Optional[str], default='CIFAR10'
        A tag to identify the dataset used in the experiment.
    model_tag : Optional[str], default='ViT'
        A tag to identify the model used in the experiment.
    experiment_directory : Optional[str], default=EXPERIMENT_PATH
        The directory where experiment results will be saved.
    """
    model: Module
    optimizer: Any
    criterion: Any
    model_params: Optional[dict] = None
    scheduler: Optional[Any] = None
    batch_size: Optional[int] = 512
    learning_rate: Optional[float] = 0.001
    epochs: Optional[int] = 100
    early_stopping_epochs: Optional[int] = 10
    pruning: Optional[bool] = False
    experiment_tag: Optional[str] = 'SWViT_CIFAR10'
    dataset_tag: Optional[str] = 'CIFAR10'
    model_tag: Optional[str] = 'ViT'
    experiment_directory: Optional[str] = EXPERIMENT_PATH

class ExperimentManager:
    """
    A class to manage and run machine learning experiments.
    
    Attributes:
    -----------
    experiment_parameters: ExperimentParameters
        The parameters for the experiment.

    Methods:
    --------
    run_experiment(train_loader, validation_loader, rounds=1, plot_results=False, save_best_model=False, save_logs=True, verbose=0):
        Runs the experiment with the given data loaders.
    """
    
    def __init__(self, experiment_parameters: ExperimentParameters) -> None:
        self.model_params = experiment_parameters.model_params
        self.model = experiment_parameters.model
        self.criterion = experiment_parameters.criterion
        self.optimizer = experiment_parameters.optimizer
        self.scheduler = experiment_parameters.scheduler
        self.experiment_directory = experiment_parameters.experiment_directory
        
        self.batch_size = experiment_parameters.batch_size
        self.learning_rate = experiment_parameters.learning_rate
        self.epochs = experiment_parameters.epochs
        self.early_stopping_epochs = experiment_parameters.early_stopping_epochs
        
        self.dataset = experiment_parameters.dataset_tag
        self.experiment_tag = '_'.join([str(experiment_parameters.experiment_tag), datetime.now().strftime("%Y%m%d%H%M")])
        self.model_tag = experiment_parameters.model_tag

        self.experiment_log = {}
        self.epoch_results = []
        self.pruning = experiment_parameters.pruning

        self.run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)
        self.run["experiment/name"] = self.experiment_tag
        self.run["experiment/dataset"] = experiment_parameters.dataset_tag
        self.run["experiment/model"] = experiment_parameters.model_tag

        self.run["experiment/learning_rate"] = experiment_parameters.learning_rate
        self.run["experiment/epochs"] = experiment_parameters.epochs
        self.run["experiment/early_stopping_epochs"] = experiment_parameters.early_stopping_epochs
        self.run["experiment/criterion"] = experiment_parameters.criterion.__name__
        self.run["experiment/optimizer"] = experiment_parameters.optimizer.__name__
        self.run["experiment/learning_rate_schedule"] = '' if experiment_parameters.scheduler is None else experiment_parameters.scheduler.__name__

    def run_experiment(self,
                    train_loader,
                    validation_loader,
                    rounds = 1,
                    plot_results = False,
                    save_best_model = False,
                    save_logs = True,
                    verbose=0):
        
        if save_logs:
            self._log_experiment_setup(rounds)
        
        for round in range(1, rounds + 1):
            
            model = self.create_model_instance()
            
            criterion = self.criterion().to(device)
            optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
            
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer, self.epochs)
            
            train_losses = []
            validation_losses = []
            train_accuracies = []
            validation_accuracies = []

            best_validation_loss = float('inf')
            best_validation_accuracy = 0
            best_model_state = None
            early_stop_counter = 0

            print(f"Experiment {self.experiment_tag} - Round {round}")
            start_experiment = datetime.now()
            for epoch in range(1, self.epochs + 1): # type: ignore
                start_training = datetime.now()
                train_loss, train_accuracy = self._train(model, criterion, train_loader, optimizer)
                end_training = datetime.now()
                
                validation_loss, validation_accuracy = self._evaluate(model, criterion, validation_loader)

                if self.scheduler is not None:
                    scheduler.step(epoch - 1)

                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                train_accuracies.append(train_accuracy)
                validation_accuracies.append(validation_accuracy)

                # Log metrics to Neptune
                self.run[f"metrics/{round}/train_loss"].log(train_loss, step=epoch)
                self.run[f"metrics/{round}/validation_loss"].log(validation_loss, step=epoch)
                self.run[f"metrics/{round}/train_accuracy"].log(train_accuracy, step=epoch)
                self.run[f"metrics/{round}/validation_accuracy"].log(validation_accuracy,step=epoch)
                
                if verbose > 0:
                    print(f"Epoch {epoch}/{self.epochs} Train Loss: {train_loss:.4f} Validation Loss: {validation_loss:.4f}")
                    print(f"Epoch {epoch}/{self.epochs} Train Accuracy: {train_accuracy:.2f}% Validation Accuracy: {validation_accuracy:.2f}%\n")
                
                if save_logs:
                    self._log_epoch_results(round, epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, training_time_minutes=(end_training - start_training).total_seconds() / 60)
                
                if validation_accuracy > best_validation_accuracy:
                    best_validation_loss = validation_loss
                    best_train_loss = train_loss
                    best_train_accuracy = train_accuracy
                    best_validation_accuracy = validation_accuracy

                    early_stop_counter = 0
                    if save_best_model:
                        best_model_state = deepcopy(model.state_dict())
                else:
                    early_stop_counter += 1

                if self.early_stopping_epochs is not None and early_stop_counter >= self.early_stopping_epochs:
                    print(f"\nEarly stopping triggered! No improvement in validation loss for {self.early_stopping_epochs} epochs.")
                    break

            end_experiment = datetime.now()
            total_time = (end_experiment - start_experiment).total_seconds() / 60
            print(f"Total experiment time: {total_time:.2f} minutes")
            print(f"Train Loss: {best_train_loss:.4f} Validation Loss: {best_validation_loss:.4f}")
            print(f"Train Accuracy: {best_train_accuracy:.2f}% Validation Accuracy: {best_validation_accuracy:.2f}%")
            print("-------------------------------------------------------\n\n\n")
            
            # Log final results to Neptune
            self.run["results/best_train_loss"] = best_train_loss
            self.run["results/best_validation_loss"] = best_validation_loss
            self.run["results/best_train_accuracy"] = best_train_accuracy
            self.run["results/best_validation_accuracy"] = best_validation_accuracy
            self.run["results/total_experiment_time_minutes"] = total_time
            
            if save_logs:
                self._log_experiment_results(model, round, total_time, epoch, best_train_loss, best_validation_loss, best_train_accuracy, best_validation_accuracy)

            if save_best_model:
                self._save_model(best_model_state, tag='best')

            if plot_results:
                epochs = range(1, len(train_losses) + 1)
                self._plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies, epochs)

        if save_logs:
            self._save_final_log()
        
        self.run.stop()
        del model, train_loader, validation_loader

    def create_model_instance(self):

        if isinstance(self.model, type):
            
            if self.model_params is None:
                model = self.model()
            elif isinstance(self.model_params,dict):
                model = self.model(**self.model_params)
            else:
                model = self.model(self.model_params)
        else:
            model = deepcopy(self.model)
        
        return model.to(device)


    def _train(self,model, criterion, train_loader, optimizer):
        train_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for batch in tqdm(train_loader, desc="Training"):
        # for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x.float())
            y_hat = y_hat.logits if hasattr(y_hat, 'logits') else y_hat  # Extract logits if the model output is a complex object
            
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy

    def _evaluate(self,model, criterion, data_loader):
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
            # for batch in data_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x.float())
                y_hat = y_hat.logits if hasattr(y_hat, 'logits') else y_hat  # Extract logits if the model output is a complex object

                loss += criterion(y_hat, y).item()

                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                y_true.extend(y.tolist())
                y_pred.extend(predicted.tolist())

        loss /= len(data_loader)
        accuracy = 100 * correct / total

        return loss, accuracy

    def _plot_results(self,train_losses, test_losses, train_accuracies, test_accuracies, epochs):
        # Plot loss curve
        plt.plot(epochs, train_losses, label='Train')
        plt.plot(epochs, test_losses, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Test Loss')
        plt.legend()
        plt.show()

        # Plot accuracy curve
        plt.plot(epochs, train_accuracies, label='Train')
        plt.plot(epochs, test_accuracies, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train and Test Accuracy')
        plt.legend()
        plt.show()

    def _log_experiment_setup(self,rounds):
        

        if self.model_params is None:
            model_params = {}
        else:
            if not(isinstance(self.model_params,dict)):
                model_params = self.model_params.to_dict()
            else:
                model_params = self.model_params
            
            model_params = {f'model_param_{k}': v for k, v in model_params.items()}

        self.experiment_log = {
            'experiment_tag':self.experiment_tag,
            'creation_timestamp':datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            'dataset':self.dataset,
            'rounds':rounds,
            'criterion':self.criterion.__name__, 
            'optimizer':self.optimizer.__name__, 
            'learning_rate':self.learning_rate,
            'learning_rate_schedule':'' if self.scheduler is None else self.scheduler.__name__ ,
            'epochs':self.epochs,
            'early_stopping_epochs':self.early_stopping_epochs,
            'model_name':self.model_tag,
            'model_trainable_params_count':0,
            **model_params,
            'experiment_results':[]
        }

        # model = self.model(**self.model_params).to(device)
        self._save_logs()
        
    def _log_epoch_results(self,round,epoch,train_loss,train_accuracy,validation_loss,validation_accuracy,training_time_minutes):
        
        epoch_results = {
            'experiment_tag':self.experiment_tag,
            'round':round,
            'epoch':epoch,
            'train_loss':train_loss,
            'train_accuracy':train_accuracy,
            'validation_loss':validation_loss,
            'validation_accuracy':validation_accuracy,
            'training_time_in_minutes':training_time_minutes
        }

        if self.epoch_results is None:
            self.epoch_results = []
        
        self.epoch_results.append(epoch_results)

    def _log_experiment_results(self,model,round,total_time,epoch,best_train_loss,best_validation_loss,best_train_accuracy,best_validation_accuracy):
        results = {
            'experiment_tag':self.experiment_tag,
            'round':round,
            'experiment_time':total_time,
            'epoch':epoch,
            'train_loss':best_train_loss,
            'validation_loss':best_validation_loss,
            'train_accuracy':best_train_accuracy,
            'validation_accuracy':best_validation_accuracy,
            'timestamp':datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            'results_per_epoch':self.epoch_results
        }
        model_trainable_params_count =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.experiment_log['model_trainable_params_count'] = model_trainable_params_count
        # self.experiment_log['model_throughput'] = model_trainable_params_count
        # self.experiment_log['model_latency'] = model_trainable_params_count

        self.experiment_log['experiment_results'].append(results)
        self._save_logs()
    
    def _save_logs(self):
        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(EXPERIMENT_PATH)
        with open(f'{path}/logs/experiment_log_{self.experiment_tag}.json', 'w') as fp:
            json.dump(self.experiment_log, fp)

    def _save_final_log(self):
        
        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(EXPERIMENT_PATH)

        experiments_setup_path = path + '/experiments_setup.csv'
        experiments_results_path = path + '/experiments_results.csv'
        experiments_epochs_path = path + '/experiments_epochs.csv'

        if os.path.exists(experiments_setup_path):
            df_experiments_setup = pd.read_csv(experiments_setup_path)
            df_experiments_setup = pd.concat([df_experiments_setup,pd.DataFrame([self.experiment_log])])
        else:
            df_experiments_setup = pd.DataFrame([self.experiment_log])
        df_experiments_setup.to_csv(experiments_setup_path,index=False)

        if os.path.exists(experiments_results_path):
            df_experiments_results = pd.read_csv(experiments_results_path)
            df_experiments_results = pd.concat([df_experiments_results,pd.DataFrame(self.experiment_log['experiment_results'])])
        else:
            df_experiments_results = pd.DataFrame(self.experiment_log['experiment_results'])
        df_experiments_results.to_csv(experiments_results_path,index=False)

        if os.path.exists(experiments_epochs_path):
            df_experiments_epochs = pd.read_csv(experiments_epochs_path)
            df_experiments_epochs = pd.concat([df_experiments_epochs,pd.DataFrame(self.epoch_results)])
        else:
            df_experiments_epochs = pd.DataFrame(self.epoch_results)
        df_experiments_epochs.to_csv(experiments_epochs_path,index=False)
        del df_experiments_epochs, df_experiments_results, df_experiments_setup
    #TODO check with lucas the following method
    def _save_model(self, model,tag=''):

        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(EXPERIMENT_PATH)

        if path is not None:
            model_save_path = os.path.join(path, f'model_{self.experiment_tag}_{tag}.pth')
            torch.save(model.state_dict(), model_save_path)
        else:
            raise ValueError("Save directory path is None")


