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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc,f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, balanced_accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps' if torch.mps.is_available() else 'cpu')

from dataclasses import dataclass
from evaluation import compute_metrics_binary, multiclass_curve

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

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
    transform_magnitude: Optional[str] = '0'
    transform_num_ops: Optional[str] = '0'
    experiment_directory: Optional[str] = '/fp/homes01/u01/ec-olathor/Documents/thesis/'
    NEPTUNE_API_TOKEN: Optional[str] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmQ2ZjBlYy04ZDQ0LTQ0ZjAtYWNhMS1hNzZlOWE0MTRmZDEifQ=="
    NEPTUNE_PROJECT: Optional[str] = "olathors-thesis/test"
    EXPERIMENT_PATH: Optional[str] = "/experiments"
    classes: Optional[int] = 2
    alpha: Optional [int] = 0
    gamma: Optional [int] = 0
    weight: Optional[Any] = None
    class_id: Optional[dict] = None


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
        self.classes = experiment_parameters.classes
        
        self.batch_size = experiment_parameters.batch_size
        self.learning_rate = experiment_parameters.learning_rate
        self.epochs = experiment_parameters.epochs
        self.early_stopping_epochs = experiment_parameters.early_stopping_epochs
        
        self.dataset = experiment_parameters.dataset_tag
        self.experiment_tag = '_'.join([str(experiment_parameters.experiment_tag), datetime.now().strftime("%Y%m%d%H%M")])
        self.model_tag = experiment_parameters.model_tag
        #self.transform_magnitude = experiment_parameters.transform_magnitude

        self.experiment_log = {}
        self.epoch_results = []
        self.class_id = experiment_parameters.class_id
        self.pruning = experiment_parameters.pruning

        self.alpha = experiment_parameters.alpha
        self.gamma = experiment_parameters.gamma
        self.weight = experiment_parameters.weight
        
        self.run = neptune.init_run(project=experiment_parameters.NEPTUNE_PROJECT, api_token=experiment_parameters.NEPTUNE_API_TOKEN)
        self.experiment_path = experiment_parameters.EXPERIMENT_PATH
        self.run["experiment/name"] = self.experiment_tag
        self.run["experiment/dataset"] = experiment_parameters.dataset_tag
        self.run["experiment/model"] = experiment_parameters.model_tag
        self.run["experiment/loss_alpha"] = experiment_parameters.alpha
        self.run["experiment/loss_gamma"] = experiment_parameters.gamma

        self.run["experiment/learning_rate"] = experiment_parameters.learning_rate
        self.run["experiment/epochs"] = experiment_parameters.epochs
        self.run["experiment/early_stopping_epochs"] = experiment_parameters.early_stopping_epochs
        self.run["experiment/criterion"] = experiment_parameters.criterion.__name__
        self.run["experiment/optimizer"] = experiment_parameters.optimizer.__name__
        self.run["experiment/learning_rate_schedule"] = '' if experiment_parameters.scheduler is None else experiment_parameters.scheduler.__name__
        self.run["experiment/transform_magnitude"] = experiment_parameters.transform_magnitude
        self.run["experiment/transform_num_ops"] = experiment_parameters.transform_num_ops
        

    def run_experiment(self,
                    train_dataset,
                    k_folds = 10,
                    rounds = 1,
                    plot_results = False,
                    save_best_model = False,
                    save_logs = True,
                    verbose=0):
        
        if save_logs:
            self._log_experiment_setup(rounds)
        
        for round in range(1, rounds + 1):
            
            model = self.create_model_instance()
            
            if self.criterion.__name__ == 'FocalLoss':
                criterion = self.criterion(alpha= self.alpha, gamma = self.gamma).to(device)
            else:
                criterion = self.criterion(weight = self.weight).to(device)

            optimizer = self.optimizer(model.parameters(), lr=self.learning_rate)
            
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer, self.epochs)
            
            train_losses = []
            validation_losses = []
            train_accuracies = []
            validation_accuracies = []
            train_aucs = []
            validation_aucs = []

            best_validation_loss = float('inf')
            best_validation_accuracy = 0
            best_validation_auc = 0
            best_model_state = None
            early_stop_counter = 0

            kf = KFold(n_splits=k_folds, shuffle=True)

            print(f"Experiment {self.experiment_tag} - Round {round}")
            start_experiment = datetime.now()
            for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):

                print(f"Fold {fold + 1}")
                print("-------")

                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                )
                validation_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(test_idx),
                )


                for epoch in range(1, self.epochs + 1): # type: ignore
                    start_training = datetime.now()
                    train_loss, train_accuracy, train_y_true, train_y_pred_proba = self._train(model, criterion, train_loader, optimizer)
                    end_training = datetime.now()
                    
                    validation_loss, validation_accuracy, val_y_true, val_y_pred_proba = self._evaluate(model, criterion, validation_loader)

                    train_metrics = compute_metrics_binary(train_y_true, train_y_pred_proba, self.classes, self.class_id)
                    validation_metrics = compute_metrics_binary(val_y_true, val_y_pred_proba, self.classes, self.class_id)

                    if self.scheduler is not None:
                        scheduler.step(epoch - 1)

                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    train_accuracies.append(train_metrics['accuracy'])
                    validation_accuracies.append(validation_metrics['accuracy'])
                    train_aucs.append(train_metrics['auc'])
                    validation_aucs.append(validation_metrics['auc'])


                    
                    
                    # Log metrics to Neptune
                    self.run[f"metrics/{round}/train_loss"].log(train_loss, step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_loss"].log(validation_loss, step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/train_accuracy"].log(train_metrics['accuracy'], step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_accuracy"].log(validation_metrics['accuracy'],step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/train_auc"].log(train_metrics['auc'], step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_auc"].log(validation_metrics['auc'],step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/train_f1score"].log(train_metrics['f1score'], step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_f1score"].log(validation_metrics['f1score'],step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/train_precision"].log(train_metrics['precision'], step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_precision"].log(validation_metrics['precision'],step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/train_recall"].log(train_metrics['recall'], step=fold*self.epochs + epoch)
                    self.run[f"metrics/{round}/validation_recall"].log(validation_metrics['recall'],step=fold*self.epochs + epoch)
                    
                    if verbose > 0:
                        print(f"Epoch {epoch}/{self.epochs} Train Loss: {train_loss:.4f} Validation Loss: {validation_loss:.4f}")
                        print(f"Epoch {epoch}/{self.epochs} Train Accuracy: {train_metrics['accuracy']:.2f}% Validation Accuracy: {validation_metrics['accuracy']:.2f}%")
                        print(f"Epoch {epoch}/{self.epochs} Train AUC: {train_metrics['auc']:.3f} Validation AUC: {validation_metrics['auc']:.3f}")
                        print(f"Epoch {epoch}/{self.epochs} Train F1 score: {train_metrics['f1score']:.3f} Validation F1 score: {validation_metrics['f1score']:.3f}")
                        print(f"Epoch {epoch}/{self.epochs} Train Precision: {train_metrics['precision']:.3f} Validation Precision: {validation_metrics['precision']:.3f}")
                        print(f"Epoch {epoch}/{self.epochs} Train Recall: {train_metrics['recall']:.3f} Validation Recall: {validation_metrics['recall']:.3f}\n")
                    
                    if save_logs:
                        self._log_epoch_results(round, epoch, train_loss, train_accuracy, validation_loss, validation_accuracy, training_time_minutes=(end_training - start_training).total_seconds() / 60)
                    
                    if validation_metrics['auc'] > best_validation_auc:
                        best_validation_loss = validation_loss
                        best_train_loss = train_loss
                        best_validation_auc = validation_metrics['auc']
                        best_train_auc = train_metrics['auc']
                        best_validation_accuracy = validation_metrics['accuracy']
                        best_train_accuracy = train_metrics['accuracy']
                        best_validation_f1score = validation_metrics['f1score']
                        best_train_f1score = train_metrics['f1score']
                        best_validation_precision = validation_metrics['precision']
                        best_train_precision = train_metrics['precision']
                        best_validation_recall = validation_metrics['recall']
                        best_train_recall = train_metrics['recall']
                        best_validation_confmat = validation_metrics['conf_mat']
                        #best_train_confmat = train_metrics['conf_mat']
                        best_validaton_roc_curve = validation_metrics['roc_auc']
                        

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
            self.run["results/best_train_auc"] = best_train_auc
            self.run["results/best_validation_auc"] = best_validation_auc
            self.run["results/best_train_f1score"] = best_train_f1score
            self.run["results/best_validation_f1score"] = best_validation_f1score
            self.run["results/best_train_precision"] = best_train_precision
            self.run["results/best_validation_precision"] = best_validation_precision
            self.run["results/best_train_recall"] = best_train_recall
            self.run["results/best_validation_recall"] = best_validation_recall
            self.run["results/total_experiment_time_minutes"] = total_time
            self.run["results/final_epoch"] = epoch

            best_validation_confmat = ConfusionMatrixDisplay(best_validation_confmat)

            fig = best_validation_confmat.plot().figure_
            
            self.run["results/best_validation_confmat"].upload(fig)

            plt.close(fig)

            y_true, y_pred_proba, threshold = best_validaton_roc_curve

            self.run["results/optimal_threshold"] = threshold

            if self.classes == 2:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                fig = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot().figure_

            else:
                fig = multiclass_curve(y_true, y_pred_proba, self.classes, self.class_id)

            self.run["results/best_roc_curve"].upload(fig)

            plt.close(fig)

            
            
            if save_logs:
                self._log_experiment_results(model, round, total_time, epoch, best_train_loss, best_validation_loss, best_train_accuracy, best_validation_accuracy)

            if save_best_model:
                self._save_model(best_model_state, tag='best')
                best_model_state = deepcopy(model.state_dict())
                self._save_model(best_model_state, tag='last')

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

        predicted_logits = torch.Tensor().to(device)
        true_labels = torch.Tensor().to(device)

        model.train()
        for batch in tqdm(train_loader, desc="Training"):
        # for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(torch.Tensor.float(x))
            y_hat = y_hat.logits if hasattr(y_hat, 'logits') else y_hat  # Extract logits if the model output is a complex object
            
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            y = y.type_as(y_hat)
            true_labels = torch.cat((true_labels,y),0)
            predicted_logits = torch.cat((predicted_logits,y_hat),0)


        predicted_probas = torch.sigmoid(predicted_logits)
        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy, true_labels, predicted_probas

    def _evaluate(self,model, criterion, data_loader):
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        predicted_logits = torch.Tensor().to(device)
        true_labels = torch.Tensor().to(device)

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
            # for batch in data_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(torch.Tensor.float(x))
                y_hat = y_hat.logits if hasattr(y_hat, 'logits') else y_hat  # Extract logits if the model output is a complex object

                loss += criterion(y_hat, y).item()

                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                y_true.extend(y.tolist())
                y_pred.extend(predicted.tolist())

                true_labels = torch.cat((true_labels,y),0)
                predicted_logits = torch.cat((predicted_logits,y_hat),0)

        predicted_probas = torch.sigmoid(predicted_logits)
        loss /= len(data_loader)
        accuracy = 100 * correct / total

        return loss, accuracy, true_labels, predicted_probas

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
        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(self.EXPERIMENT_PATH)
        with open(f'{path}/logs/experiment_log_{self.experiment_tag}.json', 'w') as fp:
            json.dump(self.experiment_log, fp)

    def _save_final_log(self):
        
        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(self.EXPERIMENT_PATH)

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

        path = os.path.expanduser(self.experiment_directory) if self.experiment_directory is not None else os.path.expanduser(self.EXPERIMENT_PATH)

        if path is not None:
            model_save_path = os.path.join(path, f'model_{self.experiment_tag}_{tag}.pth')
            torch.save(model, model_save_path)
        else:
            raise ValueError("Save directory path is None")


