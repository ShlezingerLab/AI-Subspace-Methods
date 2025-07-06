"""Subspace-Net 
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
from random import sample
from src.system_model import SystemModel, SystemModelParams
# from src.utils import *
import numpy as np
import torch

class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(noise_mean: float = 0, noise_variance: float = 1, signal_mean: float = 0,
            signal_variance: float = 1): Creates samples based on the specified mode and parameters.
        noise_creation(noise_mean, noise_variance): Creates noise based on the specified mean and variance.
        signal_creation(signal_mean=0, signal_variance=1, SNR=10): Creates signals based on the specified mode and parameters.
    """

    def __init__(self, system_model_params: SystemModelParams):
        """Initializes a Samples object.

        Args:
        -----
        system_model_params (SystemModelParams): an instance of SystemModelParams,
            containing all relevant system model parameters.

        """
        super().__init__(system_model_params)
        self.angles = None
        self.distances = None

    def set_labels(self, number_of_sources: int, angles: list, distances: list):
        if self.params.field_type.lower() == "far":
            self.set_angles(angles, number_of_sources)
        elif self.params.field_type.lower() in {"near", "full"}:
            self.set_angles(angles, number_of_sources)
            self.set_distances(distances, number_of_sources)
        else:
            raise ValueError(f"Samples.set_labels: Field type {self.params.field_type} is not defined")

    def get_labels(self):
        if self.params.field_type.lower() == "far":
            return torch.tensor(self.angles, dtype=torch.float32)
        elif self.params.field_type.lower() in {"near", "full"}:
            labels = torch.cat((torch.tensor(self.angles, dtype=torch.float32), torch.tensor(self.distances, dtype=torch.float32)), dim=0)
            return labels
        else:
            raise ValueError(f"Samples.get_labels: Field type {self.params.field_type} is not defined")


    def set_angles(self, doa: list, M: int):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """

        def create_doa_with_gap(gap: float, M: int):
            """Create angles with a value gap.

            Args:
            -----
                gap (float): Minimal gap value.

            Returns:
            --------
                np.ndarray: DOA array.

            """
            # LEGACY CODE
            # while True:
            #     # DOA = np.round(np.random.rand(M) * 180, decimals=2) - 90
            #     DOA = np.random.randint(-55, 55, M)
            #     DOA.sort()
            #     diff_angles = np.array(
            #         [np.abs(DOA[i + 1] - DOA[i]) for i in range(M - 1)]
            #     )
            #     if (np.sum(diff_angles > gap) == M - 1) and (
            #         np.sum(diff_angles < (180 - gap)) == M - 1
            #     ):
            #         break

            # based on https://stackoverflow.com/questions/51918580/python-random-list-of-numbers-in-a-range-keeping-with-a-minimum-distance
            doa_range = self.params.doa_range
            doa_resolution = self.params.doa_resolution
            if doa_resolution <= 0:
                raise ValueError("DOA resolution must be positive.")
            if M <= 0:
                raise ValueError("M (number of elements) must be positive.")
            if gap <= 0:
                raise ValueError("Gap must be positive.")

            # Compute the range of possible DOA values
            # Ensure the sampled DOAs do not exceed [-doa_range, +doa_range]
            max_offset = (gap - 1) * (M - 1)
            effective_range = 2 * doa_range - max_offset
            if effective_range <= 0:
                raise ValueError(f"Invalid effective range: {effective_range}. Check your parameters.")

            # Define the valid range for sampling
            if doa_resolution >= 1:
                valid_range = range(0, effective_range, doa_resolution)
                sampled_values = sorted(sample(valid_range, M))
            else:
                step_count = int(effective_range // doa_resolution)
                valid_range = range(step_count)
                sampled_values = sorted(sample(valid_range, M))
                sampled_values = [x * doa_resolution for x in sampled_values]

            # Compute DOAs
            DOA = [(gap - 1) * i + x - doa_range for i, x in enumerate(sampled_values)]

            # Ensure all DOAs fall naturally within the valid range
            if any(d < -doa_range or d > doa_range for d in DOA):
                raise ValueError("Computed DOAs exceed the valid range. Check your logic.")

            # Round results to 3 decimal places
            DOA = np.round(DOA, 3)

            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.angles = np.deg2rad(np.array(create_doa_with_gap(gap=10, M=M)))
        else:
            # Generate
            self.angles = np.deg2rad(doa)

    def set_distances(self, distance: list | np.ndarray, M: int) -> np.ndarray:
        """

        Args:
            distance:

        Returns:

        """

        def choose_distances(M, min_val: float, max_val: int, distance_resolution: float = 1.0) -> np.ndarray:
            """
            Choose distances for the sources.

            Args:
                M (int): Number of sources.
                min_val (float): Minimal value of the distances.
                max_val (int): Maximal value of the distances.
                distance_resolution (float, optional): Resolution of the distances. Defaults to 1.0.

            """
            distances_options = np.arange(min_val, max_val, distance_resolution)
            distances = np.random.choice(distances_options, M, replace=True)
            return np.round(distances, 3)

        if distance is None:
            self.distances = choose_distances(M, min_val=np.ceil(self.fresnel) + self.params.range_resolution,
                                              max_val=np.floor(self.fraunhofer * self.params.max_range_ratio_to_limit),
                                              distance_resolution=self.params.range_resolution)
        else:
            self.distances = np.array(distance)

    def samples_creation(
        self,
        noise_mean: float = 0,
        noise_variance: float = 1,
        signal_mean: float = 0,
        signal_variance: float = 1,
        source_number: int = None,
    ):
        """Creates samples based on the specified mode and parameters.

        Args:
        -----
            noise_mean (float, optional): Mean of the noise. Defaults to 0.
            noise_variance (float, optional): Variance of the noise. Defaults to 1.
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the signal_type is not defined.

        """
        # Generate signal matrix
        signal = self.signal_creation(signal_mean, signal_variance, source_number=source_number)
        signal = torch.from_numpy(signal)
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance)
        noise = torch.from_numpy(noise)
        if self.params.signal_type.startswith("broadband"):
            raise Exception("Samples.samples_creation: Broadband signal type is not defined for far field")
        if self.params.field_type.startswith("far"):
            A = self.steering_vec(self.angles, f_c=self.f_rng[self.params.signal_type])
            samples = (A @ signal) + noise
        elif self.params.field_type.startswith("near"):
            A = self.steering_vec(angles=self.angles, ranges=self.distances, nominal=False, generate_search_grid=False,
                                  f_c=self.f_rng[self.params.signal_type])
            samples = (A @ signal) + noise
        elif self.params.field_type.startswith("full"):
            A = self.steering_vec_full_model(angles=self.angles,
                                             ranges=self.distances)
            samples = (A @ signal) + noise
        else:
            raise Exception(f"Samples.params.field_type: Field type {self.params.field_type} is not defined")
        return samples, signal, A, noise

    def noise_creation(self, noise_mean, noise_variance):
        """Creates noise based on the specified mean and variance.

        Args:
        -----
            noise_mean (float): Mean of the noise.
            noise_variance (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for NarrowBand signal_type Noise represented in the time domain
        noise =  (
            np.sqrt(noise_variance)
            * (np.sqrt(2) / 2)
            * (
                np.random.randn(self.params.N, self.params.T)
                + 1j * np.random.randn(self.params.N, self.params.T)
            )
            + noise_mean
        )
        return noise

    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1, source_number: int = None):
        """
        Creates signals based on the specified signal nature and parameters.

        Args:
        -----
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the signal type is not defined.
            Exception: If the signal nature is not defined.
        """
        M = source_number
        if self.params.snr is None:
            snr = np.random.uniform(-5, 5)
        else:
            snr = self.params.snr
        amplitude = 10 ** (snr / 10)
        # NarrowBand signal creation
        if self.params.signal_type == "narrowband":
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                return (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(M, self.params.T)
                        + 1j * np.random.randn(M, self.params.T)
                    )
                    + signal_mean
                )

            elif self.params.signal_nature == "coherent":
                # Coherent signals: same amplitude and phase for all signals
                sig = (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(1, self.params.T)
                        + 1j * np.random.randn(1, self.params.T)
                    )
                    + signal_mean
                )
                return np.repeat(sig, M, axis=0)

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
