# Imports
import numpy as np
import torch

torch.cuda.empty_cache()
import random
import scipy
import warnings
from pathlib import Path
import pickle

from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime



def plot_spectrums(angles_grid: torch.Tensor,
                   spectrums_dict : dict,
                   true_angles: torch.Tensor = None, 
                    save: bool = False,
                    title: str = "MUSIC spectrums"
                    ):

    """    Plots the spectrums for given angles and spectrums dictionary.
    Args:
        angles_grid (torch.Tensor): A tensor containing the angles in radians.
        spectrums_dict (dict): A dictionary where keys are labels and values are tensors of spectrum values.
        true_angles (torch.Tensor, optional): A tensor containing the true angles in radians to highlight on the plot.
        save (bool, optional): If True, saves the plot as a PDF file. Defaults to False.
        title (str, optional): Title of the plot. Defaults to "MUSIC spectrums"."
    
    """
    
    angles_deg = torch.rad2deg(angles_grid).cpu().numpy()
    plt.figure(figsize=(10, 6))
    for key, value in spectrums_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"plot_spectrums: Expected torch.Tensor, got {type(value)}")
        elif len(angles_grid) != len(value):
            raise ValueError(f"plot_spectrums: Length of angles_grid ({len(angles_grid)}) "
                             f"does not match length of spectrum ({len(value)}).")
        plt.plot(angles_deg, value.cpu().detach().numpy(), 'r--', linewidth=1, label= f"{key}")
    
    
    if true_angles is not None:
        highlight_deg = torch.rad2deg(true_angles).cpu().numpy()
        for i, angle in enumerate(highlight_deg):
            plt.axvline(x=angle, color='r', linestyle='--', alpha=0.7, 
                        label='True DoA' if i == 0 else "")
    
    plt.xlabel('Angle [degrees]')
    plt.ylabel('Spectrum Power')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save:
        plt.savefig('diffmusic_spectrum.pdf')
    plt.show()

def plot_parameters(results_dict : dict):
    """Plots the array and gains for each method in the results dictionary.
    Need to also implement for refrence the nominal array and gains."""
    #TODO: implement as stated above.
    pass


def save_data_to_file(data_path: Path, *args):
    """
    Saves the provided data to a file in the specified data path.

    Args:
        data_path (Path): The path where the data will be saved.
        *args: Data to be saved.

    Returns:
        None

    """
    with open(data_path / "data.pkl", "wb") as f:
        pickle.dump(args, f)

def load_data_from_file(data_path: str):
    """
    Loads data from a file in the specified data path.
    Args:
        data_path (Path): The path from where the data will be loaded.
    Returns:
        tuple: A tuple containing the loaded data.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
    Examples:
        >>> measurements, signals, steering_mat, noise, true_angles, array, gain_impairments = load_data_from_file(Path("data"))
    """
    data_file = Path(data_path)
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    if not data:
        raise FileNotFoundError(f"load_data_from_file: No data found in {data_file}.")
    return data

def initialize_paths(main_path: Path, system_model_params) -> tuple:
    dt_string_for_save = system_model_params.dt_string_for_save
    indicating_str = (f"N:{system_model_params.N}_M:{system_model_params.M}_T:{system_model_params.T}_" + 
                      f"snr:{system_model_params.snr}_location_pert_boundary:{system_model_params.location_perturbation}_" +
                      f"gain_perturbation_var:{system_model_params.gain_perturbation_var}_" +
                      f"seed:{system_model_params.seed}")
    datasets_path = main_path / "datasets" / indicating_str / dt_string_for_save
    results_path = main_path / "results" / indicating_str / dt_string_for_save

    # create folders if not exists
    datasets_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    return datasets_path, results_path

def sample_covariance(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the sample covariance matrix for each element in the batch.

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # Add batch dimension if not present
    batch_size, sensor_number, samples_number = x.shape
    Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
    return Rx

def spatial_smoothing_covariance(x: torch.Tensor):
    """
    Calculates the covariance matrix using spatial smoothing technique for each element in the batch.

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """

    if x.dim() == 2:
        x = x[None, :, :]
    batch_size, sensor_number, samples_number = x.shape
    # Define the sub-arrays size
    sub_array_size = sensor_number // 2 + 1
    # Define the number of sub-arrays
    number_of_sub_arrays = sensor_number - sub_array_size + 1
    # Initialize covariance matrix
    Rx_smoothed = torch.zeros(batch_size, sub_array_size, sub_array_size, dtype=torch.complex128, device=device)
    Rx = sample_covariance(x)
    for j in range(number_of_sub_arrays):
        Rx_smoothed += Rx[:, j:j + sub_array_size, j:j + sub_array_size] / number_of_sub_arrays
    # Divide overall matrix by the number of sources
    return Rx_smoothed


def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)


def save_pickle(data, filepath):
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

