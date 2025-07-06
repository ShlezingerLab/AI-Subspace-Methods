"""
This file contain an implementation for a Beamformer to use for Far field(DOA) and Near field(DOA and Range) scenarios.
"""
import torch
from torch.nn import Module
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import warnings


from src.system_model import SystemModel
from src.utils import sample_covariance, keep_far_enough_points
from src.metrics import CartesianLoss, RMSPELoss
from src.config import device


class Beamformer(Module):
    def __init__(self, system_model: SystemModel):
        super(Beamformer, self).__init__()
        self.device = device
        self.system_model = system_model
        self.angles_dict = None
        self.ranges_dict = None
        self.steering_dict = None
        self.__init_grid_params()
        self.__init_steering_dict()
        self.__init_criteria()
        self.eval()  # This a torch based model without any trainable parameters.

    def forward(self, cov: torch.Tensor, sources_num: int):
        spectrum = self.get_spectrum(cov)
        peaks = self.find_peaks(spectrum, sources_num)
        labels = self.get_labels(peaks)
        return labels

    def beam_pattern(self, cov: torch.Tensor):
        """
        The MVDR beamformer implementation. it will return the optimal weights for the given input signal.
        Args:
            x:

        Returns:

        """
        eigvals, eigvecs = torch.linalg.eigh(cov)
        # inv_cov = torch.bmm(torch.bmm(eigvecs, torch.diag_embed(eigvals ** (-1)).to(torch.complex128)), eigvecs.conj().transpose(1,2))
        sqr_inv_cov = torch.bmm(torch.bmm(eigvecs, torch.diag_embed(eigvals ** (-0.5)).to(torch.complex128)), eigvecs.conj().transpose(1,2))
        # sqr_cov = torch.bmm(torch.bmm(eigvecs, torch.diag_embed(eigvals ** 0.5).to(torch.complex128)), eigvecs.conj().transpose(1,2))
        if self.system_model.params.field_type.lower() == "far":
            beam_pattern = torch.bmm(sqr_inv_cov, self.steering_dict.unsqueeze(0).repeat(cov.shape[0], 1, 1))
            beam_pattern = torch.linalg.norm(beam_pattern, dim=1)
            beam_pattern = 1 / (beam_pattern + 1e-10)
        else:
            beam_pattern = torch.einsum("bnm, mar -> bnar", sqr_inv_cov, self.steering_dict)
            beam_pattern = torch.linalg.norm(beam_pattern, dim=1)
            beam_pattern = 1 / (beam_pattern + 1e-10)
        return beam_pattern

    def plot_beam_pattern(self, beampattern: torch.Tensor, angles: torch.Tensor, ranges: torch.Tensor = None, add_title: bool = False, save: bool=False, vmin: float = None):
        """
        Plot the beam pattern of the beamformer.
        Args:
            beampattern: the beam pattern to plot.

        Returns:

        """
        if self.system_model.params.field_type.lower() == "far":
            self.plot_2D_beampattern(beampattern, angles, add_title, save, vmin)
        else:
            self.plot_3D_beampattern(beampattern, angles, ranges, add_title, save, vmin)

    def plot_2D_beampattern(self, beampattern: torch.Tensor, true_angles: torch.Tensor, add_title: bool = False, save: bool=False, vmin: float = None):
        """
        Plot the 2D beam pattern of the beamformer.
        Args:
            beampattern: the beam pattern to plot.

        Returns:

        """
        angles_rad = self.angles_dict.cpu().detach().numpy()
        beampattern = beampattern.cpu().detach().numpy()
        beampattern = beampattern / beampattern.max()
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles_rad, 10 * np.log10(beampattern), label="Beampattern")
        if true_angles is not None:
            true_angles = true_angles.cpu().detach().numpy()
            ax.plot(true_angles, np.zeros_like(true_angles), 'ro', label="True DoAs")

        # Customize the plot
        ax.set_theta_zero_location("N")  # Set 0 degrees at the top
        ax.set_theta_direction(-1)  # Clockwise direction
        ax.set_xlim(angles_rad[0], angles_rad[-1])  # Angle range
        # ax.set_ylim(-40, 10)  # dB range (adjust if needed)
        # ax.set_yticks([-40, -30, -20, -10, 0, 10])  # dB ticks

        ax.set_xlabel("Magnitude (dB)", labelpad=20)
        if add_title:
            plt.title("Beampattern of MVDR", va='bottom')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig("2D_BeamPattern.pdf")
        plt.show()

    def plot_3D_beampattern(self, beampattern: torch.Tensor, true_angles: torch.Tensor, true_ranges: torch.Tensor, add_title: bool = False, save: bool=False, vmin: float = None):
        """
        plot an heatmap of the 3D beam pattern of the beamformer.
        Args:
            beampattern:

        Returns:

        """
        angles = self.angles_dict.cpu().detach().numpy()
        ranges = self.ranges_dict.cpu().detach().numpy()
        beampattern = 10 * torch.log10(beampattern / beampattern.max())
        beampattern = beampattern.cpu().detach().numpy()
        # Create a polar plot
        theta_grid, r_grid = np.meshgrid(angles, ranges)

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        c = ax.pcolormesh(theta_grid, r_grid, beampattern.T, cmap='viridis', shading='auto', vmin=vmin)
        if true_angles is not None:
            true_angles = true_angles.cpu().detach().numpy()
            true_ranges = true_ranges.cpu().detach().numpy()
            ax.plot(true_angles, true_ranges, 'ro', label="Ground Truth")

        # Customize the plot
        plt.colorbar(c, label="Array Response (dB)")
        ax.set_theta_zero_location("E")  # Set 0 degrees at the top
        ax.set_theta_direction(-1)  # Clockwise direction
        if add_title:
            plt.title("3D Beampattern Heatmap", va='bottom')
        ax.set_xlim(angles[0], angles[-1])  # Angle range
        ax.set_ylim(ranges[0], ranges[-1])  # Range range
        ax.set_xlabel("Azimuth (rad)", labelpad=20)
        ax.set_ylabel("Range (m)", labelpad=20)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig("3D_BeamPattern.pdf")
        plt.show()

    def get_spectrum(self, cov: torch.Tensor):
        """

        Args:
            cov: the covariance matrix to use for spectrum calculation.

        Returns:
            torch.Tensor: the outcome of the beamformer for Far or Near field cases.
        """
        if self.system_model.params.field_type.lower() == "far":
            # in this case, the steering search space is 2D -> NxA, whereas A is the size of the search grid.
            v1 = torch.einsum("an, bnm -> bam", self.steering_dict.conj().transpose(0, 1), cov)
            spectrum = torch.einsum("ban, na -> ba", v1, self.steering_dict)
        else:
            # in this case, the steering search space is 3D -> NxAxR, whereas R is the size of the ranges search
            # dictionary.
            v1 = torch.einsum("arn, bnm -> barm", self.steering_dict.conj().transpose(0, 2).transpose(0, 1), cov)
            spectrum = torch.einsum("barn, nar -> bar", v1, self.steering_dict)
        if torch.imag(spectrum).abs().max() > 1e-6:
            warnings.warn(f"Beamformer.get_spectrum: Imaginary part in the spectrum is not negligible!.")
        return torch.real(spectrum)

    def find_peaks(self, spectrum: torch.Tensor, sources_num: int):
        """

        Args: spectrum: the outcome of the beamformer formula. sources_num: the number of sources to expect,
        relevant on the dataset is comprised with mix number of sources for each sample.

        Returns:
            torch.Tensor: the peaks in the spectrum to use to extract the predicted labels from the dictionary.

        """
        if self.system_model.params.field_type.lower() == "far":
            return self.__find_peaks_far_field(spectrum, sources_num)
        else:
            return self.__find_peaks_near_field(spectrum, sources_num)

    def get_labels(self, peaks: torch.Tensor):
        """

        Args:
            peaks: the indices which got peaks in the beamformer spectrum.

        Returns:
            torch.Tensor: the predicted labels for Far or Near field case.
        """
        if self.ranges_dict is None:
            # in this case, peaks is of size BxM, whereas M is the number of sources.
            labels = self.angles_dict[peaks]
        else:
            # in this case, peaks is of size Bx2M.
            peaks_angle_axis, peaks_range_axis = torch.split(peaks, peaks.shape[1] // 2, dim=1)
            angles = self.angles_dict[peaks_angle_axis]
            ranges = self.ranges_dict[peaks_range_axis]
            labels = angles, ranges
        return labels

    @staticmethod
    def pre_processing(x: torch.Tensor, mode: str = "sample"):
        """
        The pre-processing stage of the beamformer is the calculation of the sample covariance.

        Args:
            x: The input signal with dim BxNxT, for B the batch size, N the number of sensors,
                and T the number of snapshots.

        Returns:
            tensor: a Tensor of size BxNxN represent the sample covariance for each element in the batch.
        """
        if mode == "sample":
            cov = sample_covariance(x)
        else:
            raise NotImplementedError
        return cov

    def test_step(self, batch, batch_idx: int, model: Module=None):
        x, sources_num, label = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        test_length = x.shape[0]
        x = x.to(self.device)
        if max(sources_num) * 2 == label.shape[1]:
            angles, ranges = torch.split(label, max(sources_num), dim=1)
            angles = angles.to(self.device)
            ranges = ranges.to(self.device)
        else:
            angles = label.to(self.device)  # only angles

        # Check if the sources number is the same for all samples in the batch
        if (sources_num != sources_num[0]).any():
            # in this case, the sources number is not the same for all samples in the batch
            raise Exception(f"train_model:"
                            f" The sources number is not the same for all samples in the batch.")
        else:
            sources_num = sources_num[0]
        if model is not None:
            Rx = model.get_surrogate_covariance(x)
        else:
            Rx = self.pre_processing(x)
        predictions = self(Rx, sources_num)
        # self.plot_beam_pattern(self.beam_pattern(Rx)[0], angles[0], ranges[0])
        if isinstance(predictions, tuple):
            angles_prediction, ranges_prediction = predictions
            rmspe = self.criterion(angles_prediction, angles, ranges_prediction, ranges).sum(-1).item()
            _, rmspe_angle, rmspe_range = self.separated_criterion(angles_prediction, angles, ranges_prediction, ranges)
            rmspe = (rmspe, rmspe_angle.sum().item(), rmspe_range.sum().item())
        else:
            rmspe = self.criterion(predictions, angles).item()

        return rmspe, 0, test_length

    def __find_peaks_far_field(self, spectrum: torch.Tensor, known_number_of_sources):
        source_number = self.system_model.params.M
        if isinstance(source_number, tuple):
            source_number = known_number_of_sources
        batch_size = spectrum.shape[0]

        peaks = torch.zeros(batch_size, source_number, dtype=torch.int64, device=self.device)
        for batch in range(batch_size):
            elem_spectrum = spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = find_peaks(elem_spectrum, threshold=0.0)[0]
            if len(peaks_tmp) < source_number:
                warnings.warn(f"Beamformer.__find_peaks_far_field: No peaks were found! taking max values instead.")
                # random_peaks = np.random.randint(0, search_space.shape[0], (source_number - peaks_tmp.shape[0],))
                random_peaks = torch.topk(self.angles_dict, source_number - peaks_tmp.shape[0],
                                          largest=True).indices.cpu().detach().numpy()
                peaks_tmp = np.concatenate((peaks_tmp, random_peaks))
            # Sort the peak by their amplitude
            sorted_peaks = peaks_tmp[np.argsort(elem_spectrum[peaks_tmp])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[0:source_number]).to(self.device)
        # if the model is not in training mode, return the peaks
        return peaks

    def __find_peaks_near_field(self, spectrum: torch.Tensor, known_number_of_sources: int):
        batch_size = spectrum.shape[0]

        max_row = torch.zeros((batch_size, known_number_of_sources)
                              , dtype=torch.int64, device=self.device)
        max_col = torch.zeros((batch_size, known_number_of_sources)
                              , dtype=torch.int64, device=self.device)
        for batch in range(batch_size):
            elem_spectrum = spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = elem_spectrum.flatten()
            # Find spectrum peaks
            peaks = find_peaks(spectrum_flatten)[0]
            # Sort the peak by their amplitude
            sorted_peaks = peaks[np.argsort(spectrum_flatten[peaks])[::-1]]
            # convert the peaks to 2d indices
            original_idx = torch.from_numpy(np.column_stack(np.unravel_index(sorted_peaks, elem_spectrum.shape))).T
            if known_number_of_sources > 1:
                # pass
                original_idx = keep_far_enough_points(original_idx, known_number_of_sources, 10)
            max_row[batch] = original_idx[0][0: known_number_of_sources]
            max_col[batch] = original_idx[1][0: known_number_of_sources]
        # if the model is not in training mode, return the peaks.
        peaks = torch.cat((max_row, max_col), dim=1)
        return peaks

    def __init_grid_params(self):
        """
        Set values for the angle and range dict, depends on the scenario.
        Returns:
            None.

        """
        angle_range = np.deg2rad(self.system_model.params.doa_range)
        angle_resolution = np.deg2rad(self.system_model.params.doa_resolution / 2)
        # if it's the Far field case, need to init angles range.
        self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution, device=self.device,
                                        dtype=torch.float64).requires_grad_(False)
        if self.system_model.params.field_type.startswith("near"):
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            fraunhofer_ratio = self.system_model.params.max_range_ratio_to_limit
            distance_resolution = self.system_model.params.range_resolution / 2
            self.ranges_dict = torch.arange(np.ceil(fresnel),
                                            fraunhofer * fraunhofer_ratio + distance_resolution,
                                            distance_resolution,
                                            device=self.device, dtype=torch.float64).requires_grad_(False)

    def __init_steering_dict(self):
        """
        By using the dictionaries for angles and ranges, init the steering matrix to use for the beamformer.

        Returns:
            None.
        """
        if self.ranges_dict is None:
            self.steering_dict = self.system_model.steering_vec_far_field(self.angles_dict, nominal=True)
        else:
            self.steering_dict = self.system_model.steering_vec_near_field(self.angles_dict, self.ranges_dict,
                                                                nominal=True, generate_search_grid=True)

    def __init_criteria(self):
        if self.system_model.params.field_type.lower() == "far":
            self.criterion = RMSPELoss(1.0)
        else:
            self.criterion = CartesianLoss()
            self.separated_criterion = RMSPELoss(1.0)

    def __str__(self):
        return "Beamformer"
