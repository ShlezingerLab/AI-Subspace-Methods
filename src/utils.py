"""Subspace-Net 
Details
----------
Name: utils.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose:
--------
This script defines some helpful functions:
    * sum_of_diag: returns the some of each diagonal in a given matrix.
    * sum_of_diag_torch: returns the some of each diagonal in a given matrix, Pytorch oriented.
    * find_roots: solves polynomial equation defines by polynomial coefficients. 
    * find_roots_torch: solves polynomial equation defines by polynomial coefficients, Pytorch oriented.. 
    * set_unified_seed: Sets unified seed for all random attributed in the simulation.
    * get_k_angles: Retrieves the top-k angles from a prediction tensor.
    * get_k_peaks: Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.
    * gram_diagonal_overload(self, Kx: torch.Tensor, eps: float): generates Hermitian and PSD (Positive Semi-Definite) matrix,
        using gram operation and diagonal loading.
"""

# Imports
import numpy as np
import torch

torch.cuda.empty_cache()
import random
import scipy
import warnings

from pathlib import Path
from src.config import device
import matplotlib.pyplot as plt
import torch.nn as nn

# Constants
R2D = 180 / np.pi
D2R = 1 / R2D
plot_styles = {
    'CCRB': {'color': 'r', 'linestyle': '-', 'marker': 'o', "markersize": 8},
    'Beamformer': {'color': 'r', 'linestyle': '--', 'marker': 's', "markersize": 8},
    'DCD-MUSIC': {'color': 'g', 'linestyle': '-', 'marker': 'D', "markersize": 8},
    'DCD-MUSIC_V2': {'color': 'g', 'linestyle': '--', 'marker': 'd', "markersize": 8},
    'TransMUSIC': {'color': 'm', 'linestyle': '-.', 'marker': 'P', "markersize": 8},
    '2D-MUSIC': {'color': 'c', 'linestyle': ':', 'marker': '^', "markersize": 8},
    '2D-MUSIC(SPS)': {'color': 'c', 'linestyle': '--', 'marker': 'v', "markersize": 8},
    'SubspaceNet': {'color': 'k', 'linestyle': '-', 'marker': 'X', "markersize": 8},
    'NFSubspaceNet': {'color': 'k', 'linestyle': '--', 'marker': 'p', "markersize": 8},
    'NFSubspaceNet_V2': {'color': 'b', 'linestyle': '-.', 'marker': 'h', "markersize": 8},
    'ESPRIT': {'color': 'r', 'linestyle': '-', 'marker': 'v', "markersize": 8},
    'esprit(SPS)': {'color': 'r', 'linestyle': '--', 'marker': 'v', "markersize": 8},
    '1D-MUSIC': {'color': 'y', 'linestyle': '-.', 'marker': 's', "markersize": 8},
    'music(SPS)': {'color': 'y', 'linestyle': ':', 'marker': 's', "markersize": 8},
}

def validate_constant_sources_number(number_of_sources: torch.tensor):
    """
    Validate that the number of sources in the batch is equal for all samples.
    Args:
        number_of_sources: The number of sources in the batch.

    Returns:
        None

    Raises:
        ValueError: If the number of sources in the batch is not equal for all samples

    """
    if (number_of_sources != number_of_sources[0]).any():
        raise ValueError(f"validate_constant_sources_number: "
                         f"Number of sources in the batch is not equal for all samples.")

def initialize_data_paths(path: Path):
    plot_path = path / "plots"
    data_path = path / "data"
    datasets_path = data_path / "datasets"
    simulations_path = data_path / "simulations"
    saving_path = data_path / "weights"

    # create folders if not exists
    plot_path.mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)

    return datasets_path, simulations_path

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
        x = x[None, :, :]
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

def tops_covariance(x: torch.Tensor, number_of_bins: int=1):
    """
    Tops algorithm uses K bins to calculate the covariance by using STFT.
    Args:
        x:
        number_of_bins:


    Returns:

    """
    Rx = torch.zeros(x.shape[0], number_of_bins,x.shape[1], x.shape[1], dtype=torch.complex128, device=device)
    bin_size = x.shape[2] // number_of_bins
    for i in range(number_of_bins):
        x_bin = x[:, :, i*bin_size:(i+1)*bin_size]
        Rx[:, i, :, :] = sample_covariance(x_bin)
    return Rx


def keep_far_enough_points(tensor, M, D):
    # # Calculate pairwise distances between columns
    # distances = cdist(tensor.T, tensor.T, metric="euclidean")
    #
    # # Keep the first M columns as far enough points
    # selected_cols = []
    # for i in range(tensor.shape[1]):
    #     if len(selected_cols) >= M:
    #         break
    #     if all(distances[i, col] >= D for col in selected_cols):
    #         selected_cols.append(i)
    #
    # # Remove columns that are less than distance D from each other
    # filtered_tensor = tensor[:, selected_cols]
    # retrun filtered_tensor
    ##############################################
    # Extract x_coords (first dimension)
    x_coords = tensor[0, :]

    # Keep the first M columns that are far enough apart in x_coords
    selected_cols = []
    for i in range(tensor.shape[1]):
        if len(selected_cols) >= M:
            break
        if i == 0:
            selected_cols.append(i)
            continue
        if all(abs(x_coords[i] - x_coords[col]) >= D for col in selected_cols):
            selected_cols.append(i)

    # Select the columns that meet the distance criterion
    filtered_tensor = tensor[:, selected_cols]

    return filtered_tensor

# Functions
# def sum_of_diag(matrix: np.ndarray) -> list:
def sum_of_diag(matrix: np.ndarray):
    """Calculates the sum of diagonals in a square matrix.

    Args:
        matrix (np.ndarray): Square matrix for which diagonals need to be summed.

    Returns:
        list: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        >>> sum_of_diag(matrix)
        [7, 12, 15, 8, 3]

    """
    diag_sum = []
    diag_index = np.linspace(
        -matrix.shape[0] + 1,
        matrix.shape[0] + 1,
        2 * matrix.shape[0] - 1,
        endpoint=False,
        dtype=int,
    )
    for idx in diag_index:
        diag_sum.append(np.sum(matrix.diagonal(idx)))
    return diag_sum


def sum_of_diags_torch(matrix: torch.Tensor):
    """Calculates the sum of diagonals in a square matrix.
    equivalent sum_of_diag, but support Pytorch.

    Args:
        matrix (torch.Tensor): Square matrix for which diagonals need to be summed.

    Returns:
        torch.Tensor: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = torch.tensor([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])
        >>> sum_of_diag(matrix)
            torch.tensor([7, 12, 15, 8, 3])
    """
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[0] + 1, matrix.shape[0] - 1, 2 * matrix.shape[0] - 1, dtype=int
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


# def find_roots(coefficients: list) -> np.ndarray:
def find_roots(coefficients: list):
    """Finds the roots of a polynomial defined by its coefficients.

    Args:
        coefficients (list): List of polynomial coefficients in descending order of powers.

    Returns:
        np.ndarray: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = [1, -5, 6]  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        array([3., 2.])

    """
    coefficients = np.array(coefficients)
    A = np.diag(np.ones((len(coefficients) - 2,), coefficients.dtype), -1)
    if np.abs(coefficients[0]) == 0:
        A[0, :] = -coefficients[1:] / (coefficients[0] + 1e-9)
    else:
        A[0, :] = -coefficients[1:] / coefficients[0]
    roots = np.array(np.linalg.eigvals(A))
    return roots


def find_roots_torch(coefficients: torch.Tensor):
    """Finds the roots of a polynomial defined by its coefficients.
    equivalent to src.utils.find_roots, but support Pytorch.

    Args:
        coefficients (torch.Tensor): List of polynomial coefficients in descending order of powers.

    Returns:
        torch.Tensor: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = torch.tensor([1, -5, 6])  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        tensor([3., 2.])

    """
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    roots = torch.linalg.eigvals(A)
    return roots


def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)


# def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor) -> torch.Tensor:
def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k angles from a prediction tensor.

    Args:
        grid_size (float): The size of the angle grid (range) in degrees.
        k (int): The number of top angles to retrieve.
        prediction (torch.Tensor): The prediction tensor containing angle probabilities, sizeof equal to grid_size .

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angles_grid = torch.linspace(-90, 90, grid_size)
    doa_prediction = angles_grid[torch.topk(prediction.flatten(), k).indices]
    return doa_prediction


# def get_k_peaks(grid_size, k: int, prediction) -> torch.Tensor:
def get_k_peaks(grid_size: int, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.

    Args:
        grid_size (int): The size of the angle grid (range) in degrees.
        k (int): The number of top peaks (angles) to retrieve.
        prediction (torch.Tensor): The prediction tensor containing the peak values.

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angels_grid = torch.linspace(-90, 90, grid_size)
    peaks, peaks_data = scipy.signal.find_peaks(
        prediction.detach().numpy().flatten(), prominence=0.05, height=0.01
    )
    peaks = peaks[np.argsort(peaks_data["peak_heights"])[::-1]]
    doa_prediction = angels_grid[peaks]
    while doa_prediction.shape[0] < k:
        doa_prediction = torch.cat(
            (
                doa_prediction,
                torch.Tensor(np.round(np.random.rand(1) * 180, decimals=2) - 90.00),
            ),
            0,
        )

    return doa_prediction[:k]


# def gram_diagonal_overload(Kx: torch.Tensor, eps: float) -> torch.Tensor:
def gram_diagonal_overload(Kx: torch.Tensor, eps: float):
    """Multiply a matrix Kx with its Hermitian conjecture (gram matrix),
        and adds eps to the diagonal values of the matrix,
        ensuring a Hermitian and PSD (Positive Semi-Definite) matrix.

    Args:
    -----
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N],
            where BS is the batch size and N is the matrix size.
        eps (float): Constant added to each diagonal element.

    Returns:
    --------
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].

    """
    # Insuring Tensor input
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)
    Kx = Kx.to(device)

    # Kx_garm = torch.matmul(torch.transpose(Kx.conj(), 1, 2).to("cpu"), Kx.to("cpu")).to(device)
    Kx_garm = torch.bmm(Kx.conj().transpose(1, 2), Kx)
    eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[-1], device=device))).to(device)
    Kx_Out = Kx_garm + eps_addition

    # check if the matrix is Hermitian - A^H = A
    mask = (torch.abs(Kx_Out - Kx_Out.conj().transpose(1, 2)) > 1e-6)
    if mask.any():
        batch_mask = mask.any(dim=(1,2))
        warnings.warn(f"gram_diagonal_overload: {batch_mask.sum()} matrices in the batch aren't hermitian, taking the average of R and R^H.")
        Kx_Out[batch_mask] = 0.5 * (Kx_Out[batch_mask] + Kx_Out[batch_mask].conj().transpose(1, 2))

    return Kx_Out


# def _spatial_smoothing_covariance(sampels: torch.Tensor):
#     """
#     Calculates the covariance matrix using spatial smoothing technique.
#
#     Args:
#     -----
#         X (np.ndarray): Input samples matrix.
#
#     Returns:
#     --------
#         covariance_mat (np.ndarray): Covariance matrix.
#     """
#
#     X = sampels.squeeze()
#     N = X.shape[0]
#     # Define the sub-arrays size
#     sub_array_size = int(N / 2) + 1
#     # Define the number of sub-arrays
#     number_of_sub_arrays = N - sub_array_size + 1
#     # Initialize covariance matrix
#     covariance_mat = torch.zeros((sub_array_size, sub_array_size), dtype=torch.complex128)
#
#     for j in range(number_of_sub_arrays):
#         # Run over all sub-arrays
#         x_sub = X[j: j + sub_array_size, :]
#         # Calculate sample covariance matrix for each sub-array
#         sub_covariance = torch.cov(x_sub)
#         # Aggregate sub-arrays covariances
#         covariance_mat += sub_covariance / number_of_sub_arrays
#     # Divide overall matrix by the number of sources
#     return covariance_mat


def parse_loss_results_for_plotting(loss_results: dict, tested_param: str):
    plt_res = {}
    plt_acc = False
    for test, results in loss_results.items():
        for method, loss_ in results.items():
            if plt_res.get(method) is None:
                plt_res[method] = {tested_param: []}
            try:
                plt_res[method][tested_param].append(loss_[tested_param])
            except KeyError:
                plt_res[method][tested_param].append(loss_["Overall"])
            if loss_.get("Accuracy") is not None:
                if "Accuracy" not in plt_res[method].keys():
                    plt_res[method]["Accuracy"] = []
                    plt_acc = True
                plt_res[method]["Accuracy"].append(loss_["Accuracy"])
    return plt_res, plt_acc


def print_loss_results_from_simulation(loss_results: dict):
    """
    Print the loss results from the simulation.
    """
    for test, value_dict in loss_results.items():
        print("#" * 10 + f"{test} TEST RESULTS" + "#" * 10)
        for test_value, results in value_dict.items():
            if test == "SNR":
                print(f"{test} = {test_value} [dB]: ")
            else:
                print(f"{test} = {test_value}: ")
            for method, loss in results.items():
                txt = f"\t{method.upper(): <30}: "
                for key, value in loss.items():
                    if value is not None:
                        if key == "Accuracy":
                            txt += f"{key}: {value * 100:.2f} %|"
                        else:
                            txt += f"{key}: {value:.6e} |"
                print(txt)
            print("\n")
        print("\n")

class AntiRectifier(nn.Module):
    def __init__(self, relu_inplace=False):
        super(AntiRectifier, self).__init__()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), 1)
    
    def __repr__(self):
        return "AntiRectifier()"

class L2NormLayer(nn.Module):
    def __init__(self, dim=(1, 2), eps=1e-8):
        super(L2NormLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)
        # norm = torch.sqrt(torch.norm(x, p=2, dim=self.dim, keepdim=True)) + self.eps
        # return x / norm
    
    def __repr__(self):
        return f"L2NormLayer(dim={self.dim}, eps={self.eps})"

class TraceNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, Rz):
        trace = torch.real(Rz.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp(min=self.eps)  # shape [B]
        trace = trace.view(-1, 1, 1)
        return Rz / trace

    def __repr__(self):
        return f"TraceNorm(eps={self.eps})"
    
class LearnableSkipConnection(nn.Module):
    def __init__(self, alpha: float = 0.0):
        """Initializes the learnable skip connection.
        Args:
            alpha (float): Initial value for the learnable parameter.
        """
        super(LearnableSkipConnection, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)

    def forward(self, x1, x2):
        return x1 + torch.clamp(self.alpha, min=0.0, max=1.0) * x2
    
    def __repr__(self):
        return f"LearnableSkipConnection(alpha={self.alpha.item()})"

class MultipleInputIdentity(nn.Module):
    """
    A module that given multiple inputs, returns the first input as output.
    """
    def __init__(self):
        """Initializes the MultipleInputIdentity module."""
        super(MultipleInputIdentity, self).__init__()

    def forward(self, *inputs):
        return inputs[0]

    def __repr__(self):
        return "MultipleInputIdentity()"    

if __name__ == "__main__":
    # sum_of_diag example
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diag(matrix)

    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diags_torch(matrix)

    # find_roots example
    coefficients = [1, -5, 6]
    find_roots(coefficients)

    # get_k_angles example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_angles(grid_size, k, prediction)

    # get_k_peaks example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_peaks(grid_size, k, prediction)
