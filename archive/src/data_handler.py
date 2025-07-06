"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
import h5py
from torch.utils.data import Dataset, Subset
from collections import defaultdict
import os
import torch
import numpy as np
import random
from pathlib import Path

from src.signal_creation import Samples
from src.system_model import SystemModelParams

def create_dataset(
        samples_model: Samples,
        samples_size: int,
        save_datasets: bool = False,
        datasets_path: Path = None,
        true_doa: list = None,
        true_range: list = None,
        phase: str = None,
        ) -> tuple:
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        true_range (list, optional): Predefined ranges. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset and the samples model.

    """
    time_series, labels, sources_num = [], [], []

    M_is_tuple = isinstance(samples_model.params.M, tuple)

    for _ in tqdm(range(samples_size), desc="Creating dataset"):
        if M_is_tuple:
            low_M, high_M = samples_model.params.M
            high_M = min(high_M, samples_model.params.N-1)
            # make sure that low_M is less than high_M, otherwise, the randint function will raise an error
            M = low_M if low_M >= high_M else random.randint(low_M, high_M)
        else:
            M = samples_model.params.M
        # Samples model creation
        samples_model.set_labels(M, true_doa, true_range)
        # Observations matrix creation
        X = samples_model.samples_creation(
                noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1, source_number=M
            )[0]
        # Ground-truth creation
        Y = samples_model.get_labels()
        time_series.append(X)
        labels.append(Y)
        sources_num.append(M)


    generic_dataset = TimeSeriesDataset(time_series, labels, sources_num, len(set(sources_num)) == 1)
    if save_datasets:
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(samples_model.params, int(samples_size))
        generic_dataset.save(datasets_path / phase / generic_dataset_filename)

    return generic_dataset, samples_model

def load_datasets(
        system_model_params: SystemModelParams,
        samples_size: int,
        datasets_path: Path,
        is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        TimeSeriesDataset or Tuple: The desired dataset, if for test, returns the generic test dataset and the samples model.

    """
    # Generate datasets filenames
    generic_dataset = TimeSeriesDataset(None, None, None)
    model_trainingset_filename = f"Generic_DataSet" + set_dataset_filename(system_model_params, int(samples_size))
    file_name = datasets_path / f"{'train' if is_training  else 'test'}" / model_trainingset_filename
    try:
        generic_dataset.load(file_name)
        return generic_dataset
    except Exception as e:
        raise Exception(f"load_datasets: Error when loading {'Training' if is_training else 'Test'} dataset doesn't exist")


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: int):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    if isinstance(system_model_params.M, tuple):
        low_M, high_M = system_model_params.M
        M = f"random_{low_M}_{high_M}"
    else:
        M = system_model_params.M
    suffix_filename = (
            f"_{system_model_params.field_type}_field_"
            f"{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
            + f"bias={system_model_params.bias}"
            + ".h5"
    )
    return suffix_filename


class TimeSeriesDataset(Dataset):
    """
    A class for creating a dataset of time series.
    X is for the signal - a list of B elements, each element is a tensor of shape (N, T)
    Y is for the labels - a list of B elements, each element is a tensor of shape (M, ) in the far field case or (2M, ) in the near field case.
    M is for the number of sources - a list of B elements, each element is an integer.
    """
    def __init__(self, X, Y, M, is_constant_M: bool = False):
        self.X = X
        self.Y = Y
        self.M = M
        self.is_constant_M = is_constant_M
        self.path = None
        self.len = None
        self.h5f = None

    def _open_h5_file(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.path, 'r')

    def __len__(self):
        if self.len is None:
            return len(self.X)
        else:
            return self.len

    def __getitem__(self, idx):
        if self.path is None:
            return self.X[idx], self.M[idx], self.Y[idx]
        else:
            self._open_h5_file()
            x = torch.tensor(self.h5f[f'X/tensor_{idx}'][:])
            y = torch.tensor(self.h5f[f'Y/label_{idx}'][:])
            m = int(self.h5f['M'][idx])
            return x, m, y

    def get_dataloaders(self, batch_size):
        # Divide into training and validation datasets
        train_indices, val_indices = train_test_split(
            np.arange(len(self)), test_size=0.1, shuffle=True
        )
        train_dataset = Subset(self, train_indices)
        valid_dataset = Subset(self, val_indices)

        print("Training DataSet size", len(train_dataset))
        print("Validation DataSet size", len(valid_dataset))
        if os.cpu_count() > 16:
            num_workers = min(4, os.cpu_count() // 8)
        else:
            num_workers = 1
        num_workers = num_workers if num_workers > 1 else 1
        print(f"Avialble CPU cores: {os.cpu_count()}, using {num_workers}")


        if not self.is_constant_M:
            # init sampler
            batch_sampler_train = SameLengthBatchSampler(train_dataset, batch_size=batch_size)
            batch_sampler_valid = SameLengthBatchSampler(valid_dataset, batch_size=32, shuffle=False)
            # Transform datasets into DataLoader objects
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, collate_fn=collate_fn, batch_sampler=batch_sampler_train, num_workers=num_workers, worker_init_fn=worker_init_fn
            )
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, collate_fn=collate_fn, batch_sampler=batch_sampler_valid, num_workers=max(num_workers // 2, 1)
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
            )
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, shuffle=False, batch_size=32, num_workers=max(num_workers // 2, 1)
            )
        return train_dataloader, valid_dataloader

    def save(self, path):
        with h5py.File(path, 'w') as h5f:
            X_grp = h5f.create_group('X')  # Create a group for X
            for i, x in enumerate(self.X):
                X_grp.create_dataset(f"tensor_{i}", data=x.numpy())  # Save each tensor separately

            Y_grp = h5f.create_group('Y')
            for i, y in enumerate(self.Y):
                Y_grp.create_dataset(f"label_{i}", data=np.array(y))

            h5f.create_dataset('M', data=self.M)

    def load(self, path):
        self.path = path
        self._open_h5_file()
        M = self.h5f['M']
        self.len = len(M)
        self.is_constant_M = len(set(M)) == 1

        return self

    def close(self):
        """ Close the HDF5 file if it was opened """
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None

    def __del__(self):
        """ Ensure file is closed when the object is deleted """
        self.close()

def worker_init_fn(worker_id):
    """ Ensure each worker has its own HDF5 file connection. """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    # If dataset is a Subset, access the original dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    # Ensure the dataset has a path and open the HDF5 file
    if dataset.path is not None:
        dataset._open_h5_file()

def collate_fn(batch):
    """
    Collate function for the dataset loader.
    Args:
        batch:  list of tuples, each tuple contains the time series, the number of sources and the labels.

    Returns:


    """
    time_series, source_num, labels = zip(*batch)

    # Find the maximum length in this batch
    max_length = max([lb.size(0) for lb in labels])

    # Pad labels and create masks
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)

    for i, lb in enumerate(labels):
        length = lb.size(0)
        if source_num[i] != length:
            # this is a near field dataset
            angles, distances = torch.split(lb, source_num[i], dim=0)
            lb = torch.cat((angles, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
            lb = torch.cat((lb, distances, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
        else:
            lb = torch.cat((lb, torch.zeros(max_length - length, dtype=torch.long)))
        padded_labels[i] = lb

    # Stack labels
    time_series = torch.stack(time_series).squeeze()
    sources_num = torch.tensor(source_num)


    return time_series, sources_num, padded_labels


class SameLengthBatchSampler(Sampler):
    """
    A class for creating batches contains samples with the same number of sources to allow batch wise operations.

    """
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # **Preload M values efficiently**
        self.indices = np.arange(len(dataset))
        self.source_nums = np.array([dataset[i][1] for i in self.indices], dtype=int)  # Extract M values **once**

        # **Group by M values**
        self.batches = self._create_batches()

    def _create_batches(self):
        # **Sort indices by M to ensure similar-length grouping**
        grouped_by_sources = defaultdict(list)
        for idx, source_num in enumerate(self.source_nums):
            grouped_by_sources[source_num].append(idx)

        # Now, split the grouped indices into batches of batch_size
        batches = []
        for source_num, indices in grouped_by_sources.items():
            # Split indices for each source_num into batches
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])

        # **Shuffle batches (not individual samples)**
        if self.shuffle:
            np.random.shuffle(batches)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch # Convert to Python list

    def __len__(self):
        return len(self.batches)

    def get_data_source_length(self):
        return len(self.data_source)

    def get_max_batch_length(self):
        return max([len(batch) for batch in self.batches])
