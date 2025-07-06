"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import os
import time
import numpy as np
import torch.linalg
import torch.nn as nn
from sympy.vector.implicitregion import conic_coeff
from torch.utils.data.dataloader import DataLoader
from pathlib import Path

# Internal imports
from src.config import device
from src.methods_pack.music import MUSIC
from src.methods_pack.root_music import RootMusic, root_music
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.beamformer import Beamformer
from src.methods_pack.csestimator import CsEstimator
from src.models import (ModelGenerator, SubspaceNet, DCDMUSIC, DeepAugmentedMUSIC,
                        DeepCNN, DeepRootMUSIC, TransMUSIC)
from src.system_model import SystemModel, SystemModelParams


def get_model_based_method(method_name: str, system_model_params: SystemModelParams):
    """

    Parameters
    ----------
    method_name(str): the method to use - music_1d, music_2d, root_music, esprit...
    system_model(SystemModel) : the system model to use as an argument to the method class.

    Returns
    -------
    an instance of the method.
    """
    system_model = SystemModel(system_model_params, nominal=True)
    if method_name.lower().endswith("1d-music"):
        method = MUSIC(system_model=system_model, estimation_parameter="angle")
    elif method_name.lower().endswith("2d-music"):
        method = MUSIC(system_model=system_model, estimation_parameter="angle, range", model_order_estimation="aic")
    elif method_name.lower() == "root-music":
        method = RootMusic(system_model)
    elif method_name.lower().endswith("esprit"):
        method = ESPRIT(system_model)
    elif method_name.lower().endswith("beamformer"):
        method = Beamformer(system_model)
    elif method_name.lower() == "tops":
        method = TOPS(system_model)
    elif method_name.lower() == "cs_estimator":
        method = CsEstimator(system_model)
    else:
        raise NotImplementedError(f"get_model_based_method: Method {method_name} is not supported.")
    method = method.to(device)
    method = method.eval()
    return method


def get_model(params: dict, system_model_params: SystemModelParams, model_name: str = ""):
    try:
        model_name = params.get("model_name")
    except KeyError:
        pass
    model_config = (
        ModelGenerator()
        .set_model_type(model_name)
        .set_system_model(system_model_params)
        .set_model_params({x: params[x] for x in params if x != "model_name"})
        .set_model()
    )
    model = model_config.model
    path = os.path.join(Path(__file__).parent.parent, "data", "weights", model._get_name(), "final_models", model.get_model_file_name())
    try:
        model.load_state_dict(torch.load(path+".pt", map_location=device, weights_only=True))
        print(f"get_model: {model._get_name()}'s weights loaded succesfully from {path}")
    #     if isinstance(model, DCDMUSIC):
    #         model._load_state_for_angle_extractor()
    except FileNotFoundError as e:
        print("####################################")
        raise e
        # print("####################################")
        # try:
        #     print(f"Model {model_name}'s weights not found in final_models, trying to load from temp weights.")
        #     path = os.path.join(Path(__file__).parent.parent, "data", "weights", model.get_model_file_name())
        #     model.load_state_dict(torch.load(path))
        # except FileNotFoundError as e:
        #     print("####################################")
        #     print(e)
        #     print("####################################")
        #     warnings.warn(f"get_model: Model {model_name}'s weights not found")
    return model.to(device)


def evaluate_dnn_model(model: nn.Module, dataset: DataLoader, mode: str="valid") -> dict:
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (DataLoader): The evaluation dataset.

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the evaluation loss is not implemented for the model type.
    """

    # Initialize values
    overall_loss = 0.0
    overall_loss_angle = None
    overall_loss_distance = None
    overall_accuracy = None
    test_length = 0
    # if isinstance(model, DCDMUSIC):
    #     model.train_angle_extractor = False
    #     model.update_criterion()
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation
    with (torch.no_grad()):
        for idx, data in enumerate(dataset):
            if isinstance(model, (SubspaceNet, TransMUSIC, DCDMUSIC)):
                if mode == "valid":
                    eval_loss, acc = model.validation_step(data, idx)
                else:
                    eval_loss, acc = model.test_step(data, idx)
                if isinstance(eval_loss, tuple):
                    eval_loss, eval_loss_angle, eval_loss_distance = eval_loss
                    if overall_loss_angle is None:
                        overall_loss_angle, overall_loss_distance = 0.0, 0.0

                    overall_loss_angle += torch.sum(eval_loss_angle).item()
                    overall_loss_distance += torch.sum(eval_loss_distance).item()
                overall_loss += torch.sum(eval_loss).item()
                if acc is not None:
                    if overall_accuracy is None:
                        overall_accuracy = 0.0
                    overall_accuracy += acc
                if data[0].dim() == 2:
                    test_length += 1
                else:
                    test_length += data[0].shape[0]
            else:
                raise NotImplementedError(f"evaluate_dnn_model: "
                                          f"Evaluation for {model._get_name()} is not implemented yet.")
            ############################################################################################################
    overall_loss /= test_length
    if overall_loss_angle is not None and overall_loss_distance is not None:
        overall_loss_angle /= test_length
        overall_loss_distance /= test_length
    if overall_accuracy is not None:
        overall_accuracy /= test_length
    overall_loss = {"Overall": overall_loss,
                    "Angle": overall_loss_angle,
                    "Distance": overall_loss_distance,
                    "Accuracy": overall_accuracy}

    return overall_loss


def evaluate_augmented_model(augmented_method: tuple[str, str],
                            dataset,
                            system_model_params: SystemModelParams):
    """

    Args:
        augmented_method: a tuple of strings that contains the augmented method and the method to use.
        dataset: the dataset to evaluate on.
        system_model: the system model to use for the evaluation.

    Returns:
        the test score of the augmented method.
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    model_name = augmented_method[0]
    algorithm = augmented_method[1].lower()
    model_params = augmented_method[2]

    model = get_model(model_name=model_name,
                params=model_params,
                system_model_params=system_model_params)
    # Initialize instances of subspace methods
    method = get_model_based_method(algorithm, system_model_params)
    over_all_loss = 0.0
    angle_loss, distance_loss, acc = None, None, None
    test_length = 0
    if isinstance(method, nn.Module):
        method = method.to(device)
        # Set model to eval mode
        method.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if algorithm.lower() in ["1d-music", "2d-music", "esprit", "root-music", "beamformer", "tops"]:
                tmp_rmspe, tmp_acc, tmp_length = method.test_step(data, i, model)
                if isinstance(tmp_rmspe, tuple):
                    tmp_rmspe, tmp_rmspe_angle, tmp_rmspe_range = tmp_rmspe
                    if angle_loss is None:
                        angle_loss = 0.0
                    if distance_loss is None:
                        distance_loss = 0.0
                    angle_loss += tmp_rmspe_angle
                    distance_loss += tmp_rmspe_range
                over_all_loss += tmp_rmspe
                if acc is None:
                    acc = 0.0

                acc += tmp_acc
                test_length += tmp_length
            else:
                raise NotImplementedError(f"evaluate_model_based: Algorithm {algorithm} is not supported.")
        result = {"Overall": over_all_loss / test_length}
        if distance_loss is not None and angle_loss is not None:
            result["Angle"] = angle_loss / test_length
            result["Distance"] = distance_loss / test_length
        if acc is not None:
            result["Accuracy"] = acc / test_length
    return result


def evaluate_model_based(dataset: DataLoader, system_model_params: SystemModelParams, algorithm: str = "music"):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (DataLoader): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".

    Returns:
        float: The average evaluation loss.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    over_all_loss = 0.0
    angle_loss, distance_loss, acc = None, None, None
    test_length = 0
    if algorithm.lower() == "ccrb":
        if system_model_params.signal_nature.lower() == "non-coherent":
            crb = evaluate_crb(dataset, system_model_params, mode="cartesian")
            return crb
        else:
            return None
    model_based = get_model_based_method(algorithm, system_model_params)
    if isinstance(model_based, nn.Module):
        model_based = model_based.to(device)
        # Set model to eval mode
        model_based.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            tmp_rmspe, tmp_acc, tmp_length = model_based.test_step(data, i)
            if isinstance(tmp_rmspe, tuple):
                tmp_rmspe, tmp_rmspe_angle, tmp_rmspe_range = tmp_rmspe
                if angle_loss is None:
                    angle_loss = 0.0
                if distance_loss is None:
                    distance_loss = 0.0
                angle_loss += tmp_rmspe_angle
                distance_loss += tmp_rmspe_range
            over_all_loss += tmp_rmspe
            if acc is None:
                acc = 0.0

            acc += tmp_acc
            test_length += tmp_length
        # clear cache
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass
        result = {"Overall": over_all_loss / test_length}
        if distance_loss is not None and angle_loss is not None:
            result["Angle"] = angle_loss / test_length
            result["Distance"] = distance_loss / test_length
        if acc is not None:
            result["Accuracy"] = acc / test_length
    return result


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions

def evaluate_crb(dataset: DataLoader,
                 params: SystemModelParams,
                 mode: str="separate"):
    linear_snr = 10 ** (params.snr / 10)
    if params.field_type.lower() == "far":
        """
        Taken directly from the paper:
        MUSIC, Maximum Likelihood, and Cramer-Rao Bound 
        """
        ccrb = 0.0
        system_model = SystemModel(params, nominal=True)
        if params.signal_nature.lower() == "non-coherent":
            for i, data in enumerate(dataset):
                _, _, angles = data
                angles = angles.to(device)
                derivative_mat = system_model.steering_derivative(angles)
                steering_mat = system_model.steering_vec_far_field(angles, nominal=True)
                herm_steering = torch.conj(steering_mat).transpose(1, 2)
                inv_AHA = torch.linalg.inv(torch.bmm(herm_steering, steering_mat))
                projection_mat = (torch.eye(params.N, device=device).expand(angles.shape[0], params.N, params.N)
                                  - torch.bmm(steering_mat, torch.bmm(inv_AHA, herm_steering)))
                fim = (2 * params.T * linear_snr) * torch.real(torch.bmm(derivative_mat.conj().transpose(1, 2), torch.bmm(projection_mat, derivative_mat)))
                tmp_ccrb = torch.linalg.inv(fim)
                trace_ccrb = torch.sum(torch.diagonal(tmp_ccrb, dim1=1, dim2=2), dim=1)
                ccrb += torch.mean(torch.sqrt(trace_ccrb))
            ccrb /= len(dataset)
            return {"Overall": ccrb.item()}

    elif params.field_type.lower() == "near":
        """
        Taken directly from the paper:
        Conditional and Unconditional Cramér–Rao Bounds for Near-Field Source Localization
        """
        if params.signal_nature.lower() == "non-coherent":
            ccrb_angle = 0.0
            ccrb_distance = 0.0
            ccrb_cartesian = 0.0
            for i, data in enumerate(dataset):
                _, _, labels = data
                angles = labels[:, :labels.shape[1] // 2]
                distances = labels[:, labels.shape[1] // 2:]
                angles = angles.to(device)
                distances = distances.to(device)
                N = params.N
                T = params.T

                # Calculate the CCRB for the angles
                tmc_ccrb_angle = calc_angles_ccrb_near_field(angles, linear_snr, T, N, params.wavelength, params.wavelength / 2)
                ccrb_angle += torch.mean(torch.sqrt(tmc_ccrb_angle)).item()
                # Calculate the CCRB for the distances
                tmp_ccrb_distance = calc_distances_ccrb_near_field(distances, angles, linear_snr, T, N, params.wavelength, params.wavelength / 2)
                ccrb_distance += torch.mean(torch.sqrt(tmp_ccrb_distance)).item()

            if mode == "cartesian":
                # Need to calculate the cross term as well, and change coordinates.
                ccrb_cross = calc_cross_ccrb_near_field(distances, angles, linear_snr, T, N, params.wavelength, params.wavelength / 2)
                tmp_ccrb_cartesian = calc_cartesian_ccrb_near_field(tmc_ccrb_angle, tmp_ccrb_distance, ccrb_cross, angles, distances)
                ccrb_cartesian += torch.mean(torch.sqrt(tmp_ccrb_cartesian)).item()
            ccrb_angle /= len(dataset)
            ccrb_distance /= len(dataset)
            if mode == "cartesian":
                ccrb_cartesian /= len(dataset)

            return {"Overall": ccrb_cartesian,
                    "Angle": ccrb_angle,
                    "Distance": ccrb_distance}
        else:
            print("UCRB calculation for the coherent is not supported yet")
    else:
        print("Unrecognized field type.")
    return

def calc_angles_ccrb_near_field(angles, snr, T, N, wavelength, sensor_distance):
    res = 3 * wavelength ** 2
    res /= 2 * snr * T * sensor_distance ** 2 * np.pi ** 2 * torch.cos(angles) ** 2
    res *= (8 * N - 11) * (2 * N - 1)
    res /= N * (N ** 2 - 1) * (N ** 2 - 4)
    return res

def calc_distances_ccrb_near_field(distances, angles, snr, T, N, wavelength, sensor_distance):
    res = 6 * distances ** 2 * wavelength ** 2
    res /= snr * T * np.pi ** 2 * sensor_distance ** 4
    num = 15 * distances ** 2
    num += 30 * sensor_distance * distances * (N - 1) * torch.sin(angles)
    num += sensor_distance ** 2 * (8 * N - 11) * (2 * N - 1) * torch.sin(angles) ** 2
    res *= num
    res /= N ** 2 * (N ** 2 - 1) * (N ** 2 - 4) * torch.cos(angles) ** 4
    return res

def calc_cross_ccrb_near_field(distances, angles, snr, T, N, wavelength, sensor_distance):
    ccrb_cross = -3 * wavelength ** 2 * distances
    ccrb_cross /= snr * T * np.pi ** 2 * (wavelength / 2) ** 3
    ccrb_cross *= 15 * distances * (N - 1) + (wavelength / 2) * (8 * N - 11) * (2 * N - 1) * torch.sin(angles)
    ccrb_cross /= N * (N ** 2 - 1) * (N ** 2 - 4) * torch.cos(angles) ** 3
    return ccrb_cross

def calc_cartesian_ccrb_near_field(ccrb_angle, ccrb_distance, ccrb_cross, angles, distances):
    ccrb_cartesian = 0.0
    for m in range(angles.shape[-1]):
        ccrb_polar = torch.zeros(angles.shape[0], 2, 2).to(device).to(torch.float32)
        ccrb_polar[:, 0, 0] = ccrb_angle[:, m]
        ccrb_polar[:, 1, 1] = ccrb_distance[:, m]
        ccrb_polar[:, 0, 1] = ccrb_cross[:, m]
        ccrb_polar[:, 1, 0] = -ccrb_cross[:, m]
        ccrb_cartesian += transform_polar_ccrb_to_cartesian(ccrb_polar, angles[:, m], distances[:, m]) / angles.shape[
            -1]
    return ccrb_cartesian

def transform_polar_ccrb_to_cartesian(ccrb, angles, distances):
    jacobian = torch.zeros_like(ccrb)
    jacobian[:, 0, 0] = distances * torch.cos(angles)
    jacobian[:, 0, 1] = torch.sin(angles)
    jacobian[:, 1, 0] = -distances * torch.sin(angles)
    jacobian[:, 1, 1] = torch.cos(angles)
    ccrb_cartesian = torch.bmm(jacobian, torch.bmm(ccrb, jacobian.transpose(1, 2)))
    ccrb_cartesian = torch.diagonal(ccrb_cartesian, dim1=1, dim2=2).sum(-1)

    return ccrb_cartesian

def evaluate(
        generic_test_dataset: DataLoader,
        system_model_params: SystemModelParams,
        models: dict = None,
        augmented_methods: list = None,
        subspace_methods: list = None,
        model_tmp: nn.Module = None
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        generic_test_dataset (list): Test dataset for generic subspace methods.
        system_model: instance of SystemModel.
        models (dict): dict that contains the models to evluate and their parameters.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.
        model_tmp (nn.Module, optional): Temporary model for evaluation. Defaults to None.

    Returns:
        dict: Dictionary containing the evaluation results.
    """
    res = {}
    # Evaluate DNN model if given
    if model_tmp is not None:
        model_test_loss = evaluate_dnn_model(model=model_tmp, dataset=generic_test_dataset, mode="test")
        try:
            model_name = model_tmp._get_name()
        except AttributeError:
            model_name = "DNN"
        res[model_name + "_tmp"] = model_test_loss
    # Evaluate DNN models
    for model_name, params in models.items():
        model = get_model(model_name=model_name, params=params, system_model_params=system_model_params)
        # num_of_params = sum(p.numel() for p in model.parameters())
        # total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        # print(f"Number of parameters in {model_name}: {num_of_params} with total size: {total_size} bytes")
        start = time.time()
        model_test_loss = evaluate_dnn_model(model=model, dataset=generic_test_dataset, mode="test")
        print(f"{model_name} evaluation time: {time.time() - start}")
        # try:
        #     model_name = model._get_name()
        # except AttributeError:
        #     pass
        # try:
        #     model_name += f"{model.tau}"
        # except AttributeError:
        #     pass
        res[model_name] = model_test_loss
    # Evaluate SubspaceNet augmented methods
    # system_model.create_array()
    for algorithm in augmented_methods:
        loss = evaluate_augmented_model(
            augmented_method=algorithm,
            dataset=generic_test_dataset,
            system_model_params=system_model_params,
        )
        res["augmented" + f"_{algorithm[0]}_{algorithm[1]}"] = loss
    # Evaluate classical subspace methods
    # system_model.create_array()
    for algorithm in subspace_methods:
        start = time.time()
        loss = evaluate_model_based(generic_test_dataset, system_model_params, algorithm=algorithm)
        if system_model_params.signal_nature == "coherent" and algorithm.lower() in ["1d-music", "2d-music", "root-music", "esprit"]:
            algorithm += "(SPS)"
        print(f"{algorithm} evaluation time: {time.time() - start}")
        if loss is not None:
            res[algorithm] = loss
    # MLE
    # mle_loss = evaluate_mle(generic_test_dataset, system_model, criterion)
    # res["MLE"] = mle_loss
    for method, loss_ in res.items():
        cleaned_dict = {k: v for k, v in loss_.items() if v is not None}
        print(f"{method.upper() + ' test loss' : <30} = {cleaned_dict}")
    return res
