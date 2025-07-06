"""
Subspace-Net

Details
----------
Name: training.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This code provides functions for training and simulating the Subspace-Net model.

Classes:
----------
- TrainingParams: A class that encapsulates the training parameters for the model.

Methods:
----------
- train: Function for training the model.
- train_model: Function for performing the training process.
- plot_learning_curve: Function for plotting the learning curve.
- simulation_summary: Function for printing a summary of the simulation parameters.

Attributes:
----------
None
"""
# Imports
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.optim import lr_scheduler
import wandb
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.config import device
import warnings


# internal imports
from src.models import (SubspaceNet, DCDMUSIC, TransMUSIC)
from src.evaluation import evaluate_dnn_model


class TrainingParamsNew:
    def __init__(self, **kwargs):
        # Set default values
        self.learning_rate = 0.001
        self.weight_decay = 1e-9
        self.epochs = 2
        self.optimizer = "Adam"
        self.scheduler = "StepLR"
        self.step_size = 50
        self.gamma = 0.5
        self.training_objective = "angle, range"
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def update(self, new_params):
        self.__dict__.update(new_params)
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


class Trainer:
    def __init__(self, model: nn.Module, training_params: TrainingParamsNew, show_plots:bool = False):
        self.model = model
        self.training_params = training_params
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.__init_scheduler()
        self.training_objective = self.__extract_training_objective()
        self.is_wandb = False
        self.__init_paths()
        self.show_plots = show_plots
        self.device = device

    def train(self, train_dataloader, valid_dataloader, use_wandb:bool=False, save_final:bool=False,
              load_model:bool=False):
        self.model = self.model.to(self.device)
        self.__configure_model()
        self.__init_wandb(use_wandb)
        self.__load_model(load_model)
        self.__init_metrics()

        epochs = self.training_params.get("epochs", 10)
        print("\n---Start Training Stage ---\n")
        print(f"Training Objective: {self.training_objective}")
        print(f"Model: {self.model.get_model_name()}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.training_params.get('optimizer')}")
        print(f"Scheduler: {self.training_params.get('scheduler')}")
        print(f"Learning Rate: {self.training_params['learning_rate']}")
        print(f"Weight Decay: {self.training_params['weight_decay']}")
        print(f"Batch Size: {self.training_params['batch_size']}")
        print(f"Epochs: {epochs}")
        print(f"Model Checkpoint Name: {self.model.get_model_file_name()}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print("\n--- Training ---\n")

        # Run over all epochs

        since = time.time()
        for epoch in range(epochs):
            if epoch == 40:
                pass
            epoch_train_loss = 0.0
            epoch_train_loss_angle = 0.0
            epoch_train_loss_distance = 0.0
            epoch_train_acc = 0.0
            epoch_eigenregularization = 0.0
            # init tmp loss values
            # Set model to train mode
            self.model.train()
            train_length = 0


            for idx, data in tqdm(enumerate(train_dataloader), desc=f"Training {epoch + 1}/{epochs}"):
                if isinstance(self.model, (SubspaceNet, TransMUSIC, DCDMUSIC)):
                    train_loss, acc, eigen_regularization = self.model.training_step(data, idx)
                    if isinstance(train_loss, tuple):
                        train_loss, train_loss_angle, train_loss_distance = train_loss
                        epoch_train_loss_angle += train_loss_angle.item()
                        epoch_train_loss_distance += train_loss_distance.item()
                    epoch_train_loss += train_loss.item()

                    train_length += data[0].shape[0]
                    epoch_train_acc += acc
                    if eigen_regularization is not None:
                        epoch_eigenregularization += torch.sum(eigen_regularization).item()

                else:
                    raise NotImplementedError(
                        f"train_model: Training for model {self.model.get_model_name()} is not implemented")

                ############################################################################################################
                # Back-propagation stage
                try:
                    train_loss.backward(retain_graph=True)
                except RuntimeError as r:
                    raise Exception(f"linalg error: \n{r}")

                # optimizer update
                self.optimizer.step()
                # reset gradients
                self.model.zero_grad()

            ################################################################################################################
            epoch_train_loss /= train_length
            epoch_train_acc /= train_length
            epoch_eigenregularization /= train_length

            # End of epoch. Calculate the average loss
            self.loss_train_list.append(epoch_train_loss)
            if epoch_train_loss_angle != 0.0 and epoch_train_loss_distance != 0.0:
                self.loss_train_list_angles.append(epoch_train_loss_angle / train_length)
                self.loss_train_list_ranges.append(epoch_train_loss_distance / train_length)

            # Calculate evaluation loss
            valid_loss = evaluate_dnn_model(
                self.model,
                valid_dataloader,
            )

            # Calculate the average loss
            self.loss_valid_list.append(valid_loss.get("Overall"))
            self.acc_train_list.append(epoch_train_acc * 100)
            self.acc_valid_list.append(valid_loss.get('Accuracy') * 100)

            # Update schedular
            self.__schedualr_step()

            # Update eigenregularization weight
            self.__eigenregularization_step()


            self.__report_results(epoch, epoch_train_loss, epoch_train_acc, valid_loss, epoch_eigenregularization)
            self.__save_model(epoch, valid_loss)
            # Adjust the temperature of the model-based method - decrease the size of the search space.
            self.__adjust_diff_method_temperature(epoch)

        # Training complete
        time_elapsed = time.time() - since
        print("\n--- Training summary ---")
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Minimal Validation loss: {self.min_valid_loss:4f} at epoch {self.best_epoch}")
        self.__plot_res()
        self.__save_final_model(save_final)
        # close wandb connection
        self.__finish_wandb()
        return self.model

    def __report_results(self, epoch, epoch_train_loss, epoch_train_acc, valid_loss, eigenregularization=None):
        result_txt = (f"[Epoch : {epoch + 1}/{self.training_params.get('epochs', 10)}]"
                      f" Train loss = {epoch_train_loss:.6f}, Validation loss = {valid_loss.get('Overall'):.6f}")
        if valid_loss.get("Angle") is not None and valid_loss.get("Distance") is not None:
            self.loss_valid_list_angles.append(valid_loss.get("Angle"))
            self.loss_valid_list_ranges.append(valid_loss.get("Distance"))
            result_txt += f"\nAngle loss = {valid_loss.get('Angle'):.6f}, Range loss = {valid_loss.get('Distance'):.6f}"

        result_txt += (f"\nAccuracy for sources estimation: Train = {100 * epoch_train_acc:.2f}%, "
                       f"Validation = {valid_loss.get('Accuracy') * 100:.2f}%")
        result_txt += f"\nlr {self.scheduler.get_last_lr()[0]}"
        try:
            eigenregularization_weight_tmp = self.model.get_eigenregularization_weight()
        except AttributeError:
            eigenregularization_weight_tmp = None
        if eigenregularization_weight_tmp is not None:
            result_txt += f", Eigenregularization weight = {eigenregularization_weight_tmp}"
            result_txt += f", Eigenregularization = {eigenregularization}"

        print(result_txt)

        if self.is_wandb:
            wandb.log({"Train Loss": epoch_train_loss, "Validation Loss": valid_loss.get("Overall"),
                       "Accuracy Train": epoch_train_acc, "Accuracy Validation": valid_loss.get("Accuracy")})
            if self.loss_valid_list_angles and self.loss_valid_list_ranges:
                wandb.log({"Angle Loss": valid_loss.get("Angle"), "Range Loss": valid_loss.get("Distance")})

            wandb.log({"lr": self.scheduler.get_last_lr()[0]})
            if eigenregularization_weight_tmp is not None:
                wandb.log({"tmp_Eigenregularization weight": eigenregularization_weight_tmp, 
                           "Eigen Regularization" : eigenregularization})

    def __save_model(self, epoch, valid_loss):
        if self.min_valid_loss > valid_loss.get("Overall"):
            print(
                f"Validation Loss Decreased({self.min_valid_loss:.6f}--->{valid_loss.get('Overall'):.6f}) \t Saving The Model"
            )
            self.min_valid_loss = valid_loss.get("Overall")
            self.best_epoch = epoch
            # Saving State Dict
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            torch.save(self.model.state_dict(), str(self.checkpoint_path / self.model.get_model_file_name()) + ".pt")

    def __save_final_model(self, save_final: bool=True):
        if save_final:
            self.model.load_state_dict(self.best_model_wts)
            torch.save(self.model.state_dict(), str(self.final_model_checkpoint) + ".pt")
            print("Trainer.__save_final_model: Final model saved.")

    def __load_model(self, load_model: bool):
        if load_model:
            try:
                state_dict = torch.load(str(self.final_model_checkpoint) + ".pt", weights_only=True)
            except FileNotFoundError:
                print("Model not found in ", str(self.final_model_checkpoint) + ".pt")
                return None

            # don't load angle extractor weights
            if isinstance(self.model, DCDMUSIC):
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("angle_extractor")}
                model_dict = self.model.state_dict()
                model_dict.update(state_dict)
            else:
                model_dict = state_dict

            try:
                self.model.load_state_dict(model_dict)
                print("Model loaded successfully from ", str(self.final_model_checkpoint) + ".pt")
            except Exception as e:
                print("Error loading model from ", str(self.final_model_checkpoint) + ".pt")
                print(e)
                return None


    def __plot_res(self):
        now = datetime.now()
        dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

        if self.acc_train_list is not None and self.acc_valid_list is not None:
            fig_acc = plot_accuracy_curve(
                list(range(1, self.training_params.get("epochs", 10) + 1)), self.acc_train_list, self.acc_valid_list,
                model_name=self.model._get_name()
            )
            fig_acc.savefig(self.plots_path / f"Accuracy_{self.model.get_model_name()}_{dt_string_for_save}.png")
            if self.show_plots:
                fig_acc.show()
        fig_loss = plot_learning_curve(
            list(range(1, self.training_params.get("epochs", 10) + 1)), self.loss_train_list, self.loss_valid_list,
            model_name=self.model._get_name(),
            angle_train_loss=self.loss_train_list_angles,
            angle_valid_loss=self.loss_valid_list_angles,
            range_train_loss=self.loss_train_list_ranges,
            range_valid_loss=self.loss_valid_list_ranges
        )
        fig_loss.savefig(self.plots_path / f"Loss_{self.model.get_model_name()}_{dt_string_for_save}.png")
        if self.show_plots:
            fig_loss.show()
        try:
            print("over estimation = ", self.model.over_estimation_counter)
            print("under estimation = ", self.model.under_estimation_counter)
        except Exception as e:
            pass
    
    def __init_paths(self):
        self.root_path = Path(__file__).parent.parent
        self.checkpoint_path = self.root_path / "data" / "weights" / self.model._get_name()
        self.simulation_path = self.root_path / "data" / "simulations"
        self.plots_path = self.root_path / "data" / "simulations" / "plots"
        self.scores_path = self.root_path / "data" / "simulations" / "scores"
        self.final_model_checkpoint = self.checkpoint_path / "final_models" / self.model.get_model_file_name()
        self.__verify_paths()

    def __verify_paths(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.simulation_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        self.scores_path.mkdir(parents=True, exist_ok=True)
        self.final_model_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def __init_optimizer(self):
        optimizer = self.training_params.get("optimizer", "Adam")
        if optimizer == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.training_params["learning_rate"],
                              weight_decay=self.training_params["weight_decay"])
        elif optimizer == "SGD":
            return optim.SGD(self.model.parameters(), lr=self.training_params["learning_rate"],
                             weight_decay=self.training_params["weight_decay"])
        else:
            raise ValueError(f"Optimizer {optimizer} is not defined.")

    def __init_scheduler(self):
        scheduler = self.training_params.get("scheduler", "StepLR")
        if scheduler == "StepLR":
            return lr_scheduler.StepLR(self.optimizer, step_size=self.training_params["step_size"],
                                       gamma=self.training_params["gamma"])
        elif scheduler == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.training_params["gamma"],
                                                  patience=10)
        else:
            raise ValueError(f"Scheduler {scheduler} is not defined.")

    def __extract_training_objective(self):
        if self.training_params["training_objective"] == "angle":
            return "angle"
        elif self.training_params["training_objective"] == "range":
            return "range"
        elif self.training_params["training_objective"] == "angle, range":
            return "angle, range"
        elif self.training_params["training_objective"] == "source_estimation":
            return "source_estimation"
        else:
            raise ValueError(f"Training objective {self.training_params['training_objective']} is not defined.")

    def __init_wandb(self, use_wandb: bool):
        if use_wandb:
            try:
                wandb.init(entity="gast", project="dcd_music",
                           name=self.training_params.get('simulation_name'),
                           tags=[self.model._get_name()],
                           config=self.training_params,
                           allow_val_change=True)
                try:
                    wandb.config.update(self.model.system_model.params)
                except AttributeError:
                    pass
                wandb.watch(self.model, log="all")
                self.is_wandb = True
            except Exception:
                print("Error initializing wandb")

    def __finish_wandb(self):
        if self.is_wandb:
            try:
                wandb.finish(exit_code=0)
            except Exception:
                pass

    def __init_metrics(self):
        self.loss_train_list = []
        self.loss_valid_list = []
        self.loss_train_list_angles = []
        self.loss_train_list_ranges = []
        self.loss_valid_list_angles = []
        self.loss_valid_list_ranges = []
        self.acc_train_list = []
        self.acc_valid_list = []
        self.min_valid_loss = np.inf
        self.best_epoch = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def __configure_model(self):
        if isinstance(self.model, TransMUSIC):
            if self.training_objective == "source_estimation":
                transmusic_mode = "num_source_train"
            else:
                transmusic_mode = "subspace_train"
            self.model.update_train_mode(transmusic_mode)

    def __schedualr_step(self):
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self.loss_valid_list[-1])
        elif isinstance(self.scheduler, lr_scheduler.LinearLR):
            self.scheduler.step()
        else:
            warnings.warn(f"Trainer.__schedualr_step: Unknown step method for scheduler {self.scheduler}")
            try:
                self.scheduler.step()
            except Exception as e:
                print(f"Trainer.__schedualr_step: Error in scheduler step: {e}")
                raise e

    def __eigenregularization_step(self):
        try:
            self.model.update_eigenregularization_weight(self.acc_valid_list[-1])
        except AttributeError:
            pass

    def __adjust_diff_method_temperature(self, epoch):
        try:
            reset = self.model.adjust_diff_method_temperature(epoch)
            if reset:
                self.min_valid_loss = np.inf
                self.best_epoch = 0
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
        except AttributeError:
            pass


def plot_accuracy_curve(epoch_list, train_acc: list, validation_acc: list, model_name: str = None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    figure = plt.figure(figsize=(10, 6))
    title = "Learning Curve: Accuracy per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    plt.title(title)
    plt.plot(epoch_list, train_acc, label="Train")
    plt.plot(epoch_list, validation_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    return figure


def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list, model_name: str = None,
                        angle_train_loss=None, angle_valid_loss=None, range_train_loss=None, range_valid_loss=None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    title = "Learning Curve: Loss per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    if angle_train_loss and range_train_loss :

        # create 3 subplots, the main one will spread over 2 cols, and the other 2 will be under it.
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax_angle = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax_range = plt.subplot2grid((2, 2), (1, 1), colspan=1)

        ax.set_title(title)
        ax.plot(epoch_list, train_loss, label="Train")
        ax.plot(epoch_list, validation_loss, label="Validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax_angle.plot(epoch_list, angle_train_loss, label="Train")
        ax_angle.plot(epoch_list, angle_valid_loss, label="Validation")
        ax_angle.set_xlabel("Epochs")
        ax_angle.set_ylabel("Angle Loss [rad]")
        ax_angle.legend(loc="best")
        ax_range.plot(epoch_list, range_train_loss, label="Train")
        ax_range.plot(epoch_list, range_valid_loss, label="Validation")
        ax_range.set_xlabel("Epochs")
        ax_range.set_ylabel("Range Loss [m]")
        ax_range.legend(loc="best")
        # tight layout
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(epoch_list, train_loss, label="Train")
        plt.plot(epoch_list, validation_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.grid()
    return fig
