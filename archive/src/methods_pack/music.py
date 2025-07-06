import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy as sc

from src.system_model import SystemModel
from src.methods_pack.subspace_method import SubspaceMethod
from src.utils import *
from src.metrics import RMSPELoss, CartesianLoss


from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.ndimage import find_objects

def find_k_highest_peaks(matrix, k):
    """
    Find the k highest peaks in a 2D matrix using SciPy tools. A peak is defined as a
    local maximum surrounded by smaller values.

    Parameters:
    - matrix (2D array-like): Input matrix.
    - k (int): Number of highest peaks to extract.

    Returns:
    - peaks (list): List of tuples (row, col, value) representing the positions and values of the k highest peaks.
    """
    # Apply maximum filter to find local maxima
    neighborhood = maximum_filter(matrix, size=21, mode='constant', cval=-np.inf)
    local_max = (matrix == neighborhood)

    # Label the connected components of local maxima
    labeled, num_features = label(local_max)
    slices = find_objects(labeled)

    # Extract peak positions and values
    peaks = []
    try:
        for sl in slices:
            row = int((sl[0].start + sl[0].stop - 1) / 2)
            col = int((sl[1].start + sl[1].stop - 1) / 2)
            value = matrix[row, col]
            peaks.append((row, col, value))
    except Exception as e:
        pass

    # Sort peaks by value (descending) and select the top k
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:k]
    if len(peaks) < k:
        warnings.warn(f"find_k_highest_peaks: Less than {k} peaks found.")
        # add random peaks
        x_random = np.random.randint(0, matrix.shape[0], (k - len(peaks),))
        y_random = np.random.randint(0, matrix.shape[1], (k - len(peaks),))
        for i in range(k - len(peaks)):
            peaks.append((x_random[i], y_random[i], matrix[x_random[i], y_random[i]]))


    return peaks


class MUSIC(SubspaceMethod):
    """
    This is implementation of the MUSIC method for localization in Far and Near field environments.
    For Far field - only "angle" can be estimated
    For Near field - "angle", "range" and "angle, range" are the possible options.
    """

    def __init__(self, system_model: SystemModel, estimation_parameter: str, model_order_estimation: str = None):
        """

        Args:
            system_model:
            estimation_parameter:
        """
        super().__init__(system_model, model_order_estimation=model_order_estimation)
        self.estimation_params = estimation_parameter
        self.angles_dict = None
        self.ranges_dict = None
        self.steering_dict = None
        self.music_spectrum = None
        self.cell_size = None
        self.cell_size_angle = None
        self.cell_size_range = None
        self.noise_subspace = None
        self.criterion = None
        self.separated_criterion = None

        self.__init_grid_params()
        self.__init_cells(0.2)
        self.__init_criteria()
        self.__init_search_grid()

    def forward(self, cov: torch.Tensor, number_of_sources: int, known_angles=None, known_distances=None):
        """

        Args:
            cov: The covariance matrix of the input signal.
            number_of_sources: the number of sources in the signal. Needed in case the dataset comprises mix number of sources.
            known_angles: in case we are dealing with the Near field, the known angles should be passed.
            known_distances: in case we are dealing with the Near field, the known distances should be passed.

        Returns:
            tuple: the predicted parameters, the source estimation and the eigen regularization value.
        """
        # single param estimation: the search grid should be updated for each batch, else, it's the same search grid.
        if self.system_model.params.field_type in ["near", "full"] and self.estimation_params in ["range"]:
            if known_angles.shape[-1] == 1:
                self.set_search_grid(known_angles=known_angles, known_distances=known_distances)
            else:
                params = torch.zeros((cov.shape[0], number_of_sources), dtype=torch.float64, device=self.device)
                for source in range(number_of_sources):
                    params_source, _, _ = self.forward(cov, number_of_sources=number_of_sources,
                                                       known_angles=known_angles[:, source][:, None])
                    params[:, source] = params_source.squeeze()
                return params
        _, noise_subspace, source_estimation, eigen_regularization = self.subspace_separation(cov.to(torch.complex128), number_of_sources)
        inverse_spectrum = self.get_inverse_spectrum(noise_subspace.to(self.device)).to(self.device)
        if self._get_name() == "TOPS":
            self.music_spectrum = torch.sum(1 / (inverse_spectrum + 1e-10), dim=-1)
        else:
            self.music_spectrum = 1 / (inverse_spectrum + 1e-10)
        params = self.peak_finder(number_of_sources)
        return params, source_estimation, eigen_regularization

    def get_music_spectrum_from_noise_subspace(self, noise_subspace: torch.Tensor) -> torch.Tensor:
        inverse_spectrum = self.get_inverse_spectrum(noise_subspace.to(torch.complex128))
        self.music_spectrum = 1 / inverse_spectrum
        return self.music_spectrum

    def update_number_of_sensors(self, number_of_sensors: int):
        self.system_model.create_array(number_of_sensors)
        self.set_search_grid()

    def adjust_cell_size(self):
        if self.estimation_params == "range":
            if self.cell_size > 1:
                self.cell_size = int(0.8 * self.cell_size)
                if self.cell_size % 2 == 0:
                    self.cell_size -= 1
        elif self.estimation_params == "angle, range":
            if self.cell_size_angle > 1:
                self.cell_size_angle = int(0.95 * self.cell_size_angle)
                if self.cell_size_angle % 2 == 0:
                    self.cell_size_angle -= 1
            if self.cell_size_range > 1:
                self.cell_size_range = int(0.95 * self.cell_size_range)
                if self.cell_size_range % 2 == 0:
                    self.cell_size_range -= 1
        elif self.estimation_params == "angle":
            if self.cell_size > 1:
                self.cell_size = int(0.95 * self.cell_size)
                if self.cell_size % 2 == 0:
                    self.cell_size -= 1

    def get_inverse_spectrum(self, noise_subspace: torch.Tensor):
        """

        Parameters
        ----------
        noise_subspace - the noise related subspace vectors of size BatchSizex#SENSORSx(#SENSORS-#SOURCES)

        Returns
        -------
        in all cases it will return the inverse spectrum,
        in case of single param estimation it will be 1D inverse spectrum: BatchSizex(length_search_grid)
        in case of dual param estimation it will be 2D inverse spectrum:
                                                    BatchSizex(length_search_grid_angle)x(length_search_grid_distance)
        """
        # steering_dict = self.steering_dict.to(device)
        if self.system_model.params.field_type.startswith("far"):
            steering_dict = self.steering_dict[:noise_subspace.shape[1]].to(self.device)
            var1 = torch.einsum("an, bnm -> bam", steering_dict.conj().transpose(0, 1)[:, :noise_subspace.shape[1]],
                                noise_subspace)
            inverse_spectrum = torch.norm(var1, dim=2) ** 2
        else:
            if self.estimation_params.startswith("angle, range"):
                steering_dict = self.steering_dict[:noise_subspace.shape[1]].conj().transpose(0, 2).transpose(0, 1).to(self.device)
                try:
                    var1 = torch.einsum("adk, bkl -> badl",
                                        steering_dict,
                                        noise_subspace)
                    # get the norm value for each element in the batch.
                    inverse_spectrum = torch.norm(var1, dim=-1) ** 2
                except RuntimeError:
                    warnings.warn("MUSIC.get_inverse_spectrum: Out of memory error, trying to free some memory and convert the batch operation to for loop.")
                    torch.cuda.empty_cache()
                    inverse_spectrum = torch.zeros((noise_subspace.shape[0], self.angles_dict.shape[0], self.ranges_dict.shape[0]), dtype=torch.float64, device=self.device)
                    for batch in range(noise_subspace.shape[0]):
                        var1 = torch.einsum("adk, kl -> adl",
                                        steering_dict,
                                        noise_subspace[batch])
                        inverse_spectrum[batch] = torch.norm(var1, dim=-1) ** 2

                        del var1
                
            elif self.estimation_params.endswith("angle"):
                steering_dict = self.steering_dict[:noise_subspace.shape[1]].to(self.device)
                var1 = torch.einsum("an, nbm -> abm", steering_dict.conj().transpose(0, 1),
                                    noise_subspace.transpose(0, 1))
                inverse_spectrum = torch.norm(var1, dim=-1).T ** 2
            elif self.estimation_params.startswith("range"):
                steering_dict = self.steering_dict[:noise_subspace.shape[1]].to(device)
                var1 = torch.bmm(steering_dict.conj().transpose(0, 2).transpose(0, 1), noise_subspace)
                inverse_spectrum = torch.norm(var1, dim=-1) ** 2
                if torch.isnan(inverse_spectrum).any():
                    raise ValueError("Nan values in inverse spectrum")
            else:
                raise ValueError(f"MUSIC.get_inverse_spectrum: unknown estimation param {self.estimation_params}")
        del steering_dict
        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass
        return inverse_spectrum

    def peak_finder(self, source_number: int):
        """

        Parameters
        ----------
        is_soft: this boolean paramter will determine wether to use derivative approxamtion of the peak_finder for
         the training stage.

        Returns
        -------
        the predicted param(torch.Tensor) or params(tuple)
        """
        if self.system_model.params.field_type.lower().startswith("far"):
            return self._peak_finder_1d(self.angles_dict, source_number)
        else:
            if self.estimation_params.startswith("angle, range"):
                return self._peak_finder_2d(source_number)
            elif self.estimation_params.endswith("angle"):
                return self._peak_finder_1d(self.angles_dict, source_number)
            elif self.estimation_params.startswith("range"):
                return self._peak_finder_1d(self.ranges_dict, source_number)

    def set_search_grid(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        if self.system_model.params.field_type.startswith("far"):
            self.__set_search_grid_far_field()
        elif self.system_model.params.field_type in ["near", "full"]:
            self.__set_search_grid_near_field(known_angles=known_angles, known_distances=known_distances)
        else:
            raise ValueError(f"MUSIC.set_search_grid: Unrecognized field type: {self.system_model.params.field_type}")



    def plot_spectrum(self, highlight_corrdinates=None, batch: int = 0, method: str = "heatmap", music_spectrum = None, add_title: bool = False, save: bool = False):
        if self.estimation_params == "angle, range":
            self._plot_3d_spectrum(highlight_corrdinates, batch, method, music_spectrum=music_spectrum, add_title=add_title, save=save)
        else:
            self._plot_1d_spectrum(highlight_corrdinates, batch, add_title=add_title, save=save)

    def test_step(self, batch, batch_idx, model: nn.Module=None):
        x, sources_num, label = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        test_length = x.shape[0]
        x = x.to(self.device)
        if self.estimation_params == "angle, range":
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
            try:
                Rx = model.get_surrogate_covariance(x)
            except NotImplementedError as e:
                raise e
        else:
            if self.system_model.params.signal_nature == "non-coherent":
                Rx = self.pre_processing(x, mode="sample")
            else:
                # Rx = self.pre_processing(x, mode="sps")
                Rx = self.pre_processing(x, mode="sample")
        predictions, sources_num_estimation, _ = self(Rx, number_of_sources=sources_num)
        if self.estimation_params == "angle, range":
            angles_prediction, ranges_prediction = predictions
            rmspe = self.criterion(angles_prediction, angles, ranges_prediction, ranges).sum(-1).item()
            _, rmspe_angle, rmspe_range = self.separated_criterion(angles_prediction, angles, ranges_prediction, ranges)
            rmspe = (rmspe, rmspe_angle.sum(-1).item(), rmspe_range.sum(-1).item())
        else:
            rmspe = self.criterion(predictions, angles).sum().item()

        acc = self.source_estimation_accuracy(sources_num, sources_num_estimation)

        return rmspe, acc, test_length

    def _peak_finder_1d(self, search_space, source_number: int):
        if self.estimation_params == "range":
            source_number = 1  # for the range estimation, only one source is expected.

        batch_size = self.music_spectrum.shape[0]

        peaks = torch.zeros(batch_size, source_number, dtype=torch.int64, device=self.device)
        for batch in range(batch_size):
            music_spectrum = self.music_spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = sc.signal.find_peaks(music_spectrum, threshold=0.0)[0]
            if len(peaks_tmp) < source_number:
                warnings.warn(f"MUSIC._peak_finder_1d: No peaks were found! taking max values instead.")
                # random_peaks = np.random.randint(0, search_space.shape[0], (source_number - peaks_tmp.shape[0],))
                random_peaks = torch.topk(torch.from_numpy(music_spectrum), source_number - peaks_tmp.shape[0],
                                          largest=True).indices.cpu().detach().numpy()
                peaks_tmp = np.concatenate((peaks_tmp, random_peaks))
            # Sort the peak by their amplitude
            sorted_peaks = peaks_tmp[np.argsort(music_spectrum[peaks_tmp])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[0:source_number]).to(self.device)
        if not self.training:
            # if the model is not in training mode, return the peaks
            if peaks.dim() == 1:
                return search_space[peaks]
            else:
                labels = torch.gather(search_space.unsqueeze(1).repeat(1, source_number).to(self.device), 0, peaks)
                return labels
        else:
            return self.__maskpeak_1d(peaks, search_space, source_number)

    def _peak_finder_2d(self, source_number: int):
        batch_size = self.music_spectrum.shape[0]

        max_row = torch.zeros((batch_size, source_number)
                              , dtype=torch.int64, device=self.device)
        max_col = torch.zeros((batch_size, source_number)
                              , dtype=torch.int64, device=self.device)
        for batch in range(batch_size):
            music_spectrum = self.music_spectrum[batch].detach().cpu().numpy().squeeze()
            # # Flatten the spectrum
            # spectrum_flatten = music_spectrum.flatten()
            # # Find spectrum peaks
            # peaks = sc.signal.find_peaks(spectrum_flatten)[0]
            # # Sort the peak by their amplitude
            # sorted_peaks = peaks[np.argsort(spectrum_flatten[peaks])[::-1]]
            # if len(sorted_peaks) < source_number:
            #     warnings.warn(f"MUSIC._peak_finder_2d: No peaks were found! taking max values instead.")
            #     max_peaks = torch.topk(torch.from_numpy(spectrum_flatten), source_number - sorted_peaks.shape[0],
            #                               largest=True).indices.cpu().detach().numpy()
            #     sorted_peaks = np.concatenate((sorted_peaks, max_peaks))
            # # convert the peaks to 2d indices
            # original_idx = torch.from_numpy(np.column_stack(np.unravel_index(sorted_peaks, music_spectrum.shape))).T
            # if source_number > 1:
            #     # pass
            #     original_idx = keep_far_enough_points(original_idx, source_number, 10)
            peaks = find_k_highest_peaks(music_spectrum, source_number)
            original_idx = torch.from_numpy(np.array(peaks)[:, :2]).T
            max_row[batch] = original_idx[0][0: source_number]
            max_col[batch] = original_idx[1][0: source_number]
        if not self.training:
            # if the model is not in training mode, return the peaks.
            angle_dict = self.angles_dict.to(self.device)
            range_dict = self.ranges_dict.to(self.device)
            angles_pred = angle_dict[max_row]
            distances_pred = range_dict[max_col]
            del angle_dict, range_dict
            try:
                torch.cuda.empty_cache()
            except AttributeError:
                pass
            return angles_pred, distances_pred
        else:
            return self.__maskpeak_2d(max_row, max_col, source_number)

    def __maskpeak_1d(self, peaks, search_space, source_number: int = None):

        batch_size = self.music_spectrum.shape[0]
        soft_decision = torch.zeros(batch_size, source_number, dtype=torch.float64, device=self.device)
        top_indxs = peaks.to(self.device)

        for source in range(source_number):
            cell_idx = (top_indxs[:, source][:, None]
                        - self.cell_size
                        + torch.arange(2 * self.cell_size + 1, dtype=torch.long, device=self.device))
            # cell_idx %= self.music_spectrum.shape[1]
            out_of_bounds_mask = (cell_idx < 0) | (cell_idx >= self.music_spectrum.shape[1])
            cell_idx[out_of_bounds_mask] = top_indxs[:, source].unsqueeze(1).expand_as(cell_idx)[out_of_bounds_mask]
            cell_idx = cell_idx.reshape(batch_size, -1, 1)
            metrix_thr = torch.gather(self.music_spectrum.unsqueeze(-1).expand(-1, -1, cell_idx.size(-1)), 1,
                                      cell_idx).requires_grad_(True)
            soft_max = torch.softmax(metrix_thr, dim=1)
            soft_decision[:, source][:, None] = torch.einsum("bms, bms -> bs", search_space[cell_idx.cpu()].to(self.device), soft_max).to(
                self.device)

        return soft_decision

    def __maskpeak_2d(self, peaks_r, peaks_c, source_number):
        batch_size = self.music_spectrum.shape[0]
        soft_row = torch.zeros((batch_size, source_number), device=self.device)
        soft_col = torch.zeros((batch_size, source_number), device=self.device)

        for source in range(source_number):
            max_row_cell_idx = (peaks_r[:, source][:, None]
                                - self.cell_size_angle
                                + torch.arange(2 * self.cell_size_angle + 1, dtype=torch.int32, device=self.device))
            max_row_cell_idx %= self.music_spectrum.shape[1]
            max_row_cell_idx = max_row_cell_idx.reshape(batch_size, -1, 1)

            max_col_cell_idx = (peaks_c[:, source][:, None]
                                - self.cell_size_range
                                + torch.arange(2 * self.cell_size_range + 1, dtype=torch.int32, device=self.device))
            max_col_cell_idx %= self.music_spectrum.shape[2]
            max_col_cell_idx = max_col_cell_idx.reshape(batch_size, 1, -1)

            metrix_thr = self.music_spectrum.gather(1,
                                                    max_row_cell_idx.expand(-1, -1, self.music_spectrum.shape[2]))
            metrix_thr = metrix_thr.gather(2, max_col_cell_idx.repeat(1, max_row_cell_idx.shape[-2], 1))
            soft_max = torch.softmax(metrix_thr.view(batch_size, -1), dim=1).reshape(metrix_thr.shape)
            soft_row[:, source][:, None] = torch.einsum("bla, bad -> bl",
                                                        self.angles_dict[max_row_cell_idx].transpose(1, 2),
                                                        torch.sum(soft_max, dim=2).unsqueeze(-1))
            soft_col[:, source][:, None] = torch.einsum("bmc, bcm -> bm",
                                                        self.ranges_dict[max_col_cell_idx],
                                                        torch.sum(soft_max, dim=1).unsqueeze(-1))

        return soft_row, soft_col

    def _init_spectrum(self, batch_size):
        if self.system_model.params.field_type == "Far":
            self.music_spectrum = torch.zeros(batch_size, len(self.angles_dict))
        else:
            if self.estimation_params.startswith("angle, range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angles_dict), len(self.ranges_dict))
            elif self.estimation_params.endswith("angle"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angles_dict))
            elif self.estimation_params.startswith("range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.ranges_dict))

    def __init_grid_params(self):
        angle_range = np.deg2rad(self.system_model.params.doa_range)
        angle_resolution = np.deg2rad(self.system_model.params.doa_resolution / 2)
        angle_decimals = int(np.ceil(np.log10(1 / angle_resolution)))

        if self.system_model.params.field_type.startswith("far"):
            # if it's the Far field case, need to init angles range.
            self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution,
                                            dtype=torch.float64).to(torch.float64)
            self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)
        elif self.system_model.params.field_type in ["near", "full"]:
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            if self.estimation_params.startswith("angle"):
                self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution,
                                                dtype=torch.float64).to(torch.float64)
                # self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)


            if self.estimation_params.endswith("range"):
                fraunhofer_ratio = self.system_model.params.max_range_ratio_to_limit
                distance_resolution = self.system_model.params.range_resolution / 2
                max_distance = min(self.system_model.fraunhofer, fraunhofer * fraunhofer_ratio + distance_resolution)
                self.ranges_dict = torch.arange(np.ceil(fresnel),
                                                max_distance,
                                                distance_resolution, dtype=torch.float64)
        else:
            raise ValueError(f"MUSIC.__define_grid_params: Unrecognized field type for MUSIC class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")

    def __init_search_grid(self):
        # if this is the music 2D case, the search grid is constant and can be calculated once.
        if self.system_model.params.field_type in ["near", "full"]:
            if self.angles_dict is not None and self.ranges_dict is not None:
                self.set_search_grid()
            elif self.angles_dict is not None:  # Near field case with Far field inference
                self.__set_search_grid_far_field()
        else:
            self.set_search_grid()

    def __init_cells(self, coeff: float = 0.1):

        if self.estimation_params == "range":
            self.cell_size = int(self.ranges_dict.shape[0] * coeff)
        elif self.estimation_params == "angle":
            self.cell_size = int(self.angles_dict.shape[0] * coeff)
        elif self.estimation_params == "angle, range":
            self.cell_size_angle = int(self.angles_dict.shape[0] * coeff)
            self.cell_size_range = int(self.ranges_dict.shape[0] * coeff)

        if self.cell_size is not None:
            if self.cell_size % 2 == 0:
                self.cell_size += 1
        if self.cell_size_angle is not None:
            if self.cell_size_angle % 2 == 0:
                self.cell_size_angle += 1
        if self.cell_size_range is not None:
            if self.cell_size_range % 2 == 0:
                self.cell_size_range += 1

    def init_cells(self, coeff: float = 0.2):
        self.__init_cells(coeff)

    def _plot_1d_spectrum(self, highlight_corrdinates, batch, add_title: bool = False, save: bool = False):
        if self.estimation_params == "angle":
            x = np.rad2deg(self.angles_dict.detach().cpu().numpy())
            x_label = "angle [deg]"
        elif self.estimation_params == "range":
            x = self.ranges_dict.detach().cpu().numpy()
            x_label = "distance [m]"
        else:
            raise ValueError(f"MUSIC._plot_1d_spectrum: No such option for param estimation.")
        y = self.music_spectrum[batch].detach().cpu().numpy()
        plt.figure()
        plt.plot(x, y.T, label="Music Spectrum")
        if highlight_corrdinates is not None:
            for idx, dot in enumerate(highlight_corrdinates):
                plt.vlines(dot, np.min(y), np.max(y), colors='r', linestyles='dashed', label=f"Ground Truth")
        if add_title:
            plt.title("MUSIC SPECTRUM")
        plt.grid()
        plt.ylabel("Spectrum power")
        plt.xlabel(x_label)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig("1d_music_spectrum.pdf")
        plt.show()

    def _plot_3d_spectrum(self, highlight_coordinates, batch, method, music_spectrum=None, add_title: bool = False, save: bool = False):
        """
        Plot the MUSIC 2D spectrum.

        """
        if method == "3D":
            # Creating figure
            distances = self.ranges_dict.detach().cpu().numpy()
            angles = self.angles_dict.detach().cpu().numpy()
            if music_spectrum is None:
                spectrum = self.music_spectrum[batch].detach().cpu().numpy()
            else:
                spectrum = music_spectrum[batch].detach().cpu().numpy()
            x, y = np.meshgrid(distances, np.rad2deg(angles))
            # Plotting the 3D surface
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis')

            if highlight_coordinates:
                highlight_coordinates = np.array(highlight_coordinates)
                ax.scatter(
                    highlight_coordinates[:, 0],
                    np.rad2deg(highlight_coordinates[:, 1]),
                    np.log1p(highlight_coordinates[:, 2]),
                    color='red',
                    s=50,
                    label='Ground Truth',
                    marker="x"
                )
            if add_title:
                ax.set_title('MUSIC spectrum')
            ax.set_xlim(distances[0], distances[-1])
            ax.set_ylim(np.rad2deg(angles[0]), np.rad2deg(angles[-1]))
            # Adding labels
            ax.set_ylabel('Theta [deg]')
            ax.set_xlabel('Radius [m]')
            ax.set_zlabel('Power [dB]')
            plt.colorbar(ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis'), shrink=0.5, aspect=5)

            if highlight_coordinates:
                ax.legend()  # Adding a legend

            # Display the plot
            plt.tight_layout()
            if save:
                plt.savefig("3d_music_spectrum.pdf")
            plt.show()
        elif method == "heatmap":
            xmin, xmax = np.min(self.ranges_dict.cpu().detach().numpy()), np.max(self.ranges_dict.cpu().detach().numpy())
            ymin, ymax = np.min(self.angles_dict.cpu().detach().numpy()), np.max(self.angles_dict.cpu().detach().numpy())
            if music_spectrum is None:
                spectrum = self.music_spectrum[batch].cpu().detach().numpy()
            else:
                spectrum = music_spectrum[batch].cpu().detach().numpy()
            plt.imshow(spectrum, cmap="hot",
                       extent=[xmin, xmax, np.rad2deg(ymin), np.rad2deg(ymax)], origin='lower', aspect="auto")
            if highlight_coordinates is not None:
                for idx, dot in enumerate(highlight_coordinates):
                    x = self.ranges_dict.cpu().detach().numpy()[dot[1]]
                    y = np.rad2deg(self.angles_dict.cpu().detach().numpy()[dot[0]])
                    plt.plot(x, y, label=f"{x:.1f} [m], {y:.1f} [deg]", marker='o', markerfacecolor='none',
                             markeredgecolor='white', linestyle='-', color='white', markersize=10)
                    # plt.plot(x, y, marker='x', linestyle='', color='green', markersize=8)
                plt.legend()
            plt.colorbar()
            if add_title:
                plt.title("MUSIC Spectrum heatmap")
            plt.xlabel("Distances [m]")
            plt.ylabel("Angles [deg]")

            plt.figaspect(2)
            plt.tight_layout()
            if save:
                plt.savefig("heatmap_music_spectrum.pdf")
            plt.show()
        elif method == "slice":
            x = self.ranges_dict.detach().cpu().numpy()
            x_label = "distance [m]"
            y = self.music_spectrum[batch].detach().cpu().numpy()[highlight_coordinates[0]]
            plt.figure()
            plt.plot(x, y.T, label="Music Spectrum")
            if highlight_coordinates is not None:
                for idx, dot in enumerate(highlight_coordinates[1:]):
                    plt.vlines(dot, np.min(y), np.max(y), colors='r', linestyles='dashed', label=f"Ground Truth")
            if add_title:
                plt.title(f"MUSIC SPECTRUM Slice at {torch.round(torch.rad2deg(self.angles_dict[highlight_coordinates[0]]))}")
            plt.grid()
            plt.ylabel("Spectrum power")
            plt.xlabel(x_label)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig("slice_music_spectrum.pdf")
            plt.show()

    def __set_search_grid_far_field(self):
        self.steering_dict = self.system_model.steering_vec_far_field(self.angles_dict, f_c=None, nominal=True, fix_sv_noise=True).squeeze(-1)

    def __set_search_grid_near_field(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        """

        Returns:

        """
        if known_angles is None:
            known_angles = self.angles_dict
        if known_distances is None:
            known_distances = self.ranges_dict
        self.steering_dict = self.system_model.steering_vec_near_field(angles=known_angles, ranges=known_distances,
                                                                       generate_search_grid=True, nominal=True,
                                                            f_c=None).squeeze(-1).cpu()
        if torch.isnan(self.steering_dict).any():
            raise ValueError("Nan values in steering matrix")

    def __str__(self):
        if self.estimation_params == "angle":
            return "music_angle"
        elif self.estimation_params == "range":
            return "music_range"
        elif self.estimation_params == "angle, range":
            return "2d_music"

    def __init_criteria(self):
        if self.estimation_params == "angle":
            self.criterion = RMSPELoss(balance_factor=1.0)
        elif self.estimation_params == "range":
            self.criterion = RMSPELoss(balance_factor=0.0)
        elif self.estimation_params == "angle, range":
            self.criterion = CartesianLoss()
            self.separated_criterion = RMSPELoss(1.0)
        else:
            raise ValueError(f"MUSIC.__init_criteria: Unrecognized estimation param {self.estimation_params}")





class Filter(nn.Module):
    def __init__(self, min_cell_size, max_cell_size, number_of_filter=10):
        super(Filter, self).__init__()
        self.number_of_filters = number_of_filter
        self.cell_sizes = torch.linspace(min_cell_size, max_cell_size, number_of_filter).to(torch.int32).to(self.device)
        self.cell_bank = {}
        for cell_size in enumerate(self.cell_sizes.data):
            cell_size = cell_size[1]
            self.cell_bank[cell_size] = torch.arange(-cell_size, cell_size, 1, dtype=torch.long, device=self.device)
        self.fc = nn.Linear(self.number_of_filters, 1)
        self.fc.weight.data = torch.randn(1, number_of_filter) / 100 + (1 / number_of_filter)
        self.fc.weight.data = self.fc.weight.data.to(torch.float64)
        self.fc.bias.data = torch.Tensor([0])
        self.fc.bias.data = self.fc.bias.data.to(torch.float64)
        self.fc.bias.requires_grad_(False)
        self.relu = nn.ReLU()

    def forward(self, input, search_space):
        peaks = torch.zeros(input.shape[0], 1).to(torch.int64)
        for batch in range(peaks.shape[0]):
            music_spectrum = input[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks_tmp.sort(key=lambda x: music_spectrum[x], reverse=True)
            if len(peaks_tmp) == 0:
                peaks_tmp = torch.randint(search_space.shape[0], (1,))
            else:
                peaks_tmp = peaks_tmp[0]
            peaks[batch] = peaks_tmp
        top_1 = peaks
        output = torch.zeros(input.shape[0], self.number_of_filters).to(self.device).to(torch.float64)
        for idx, cell in enumerate(self.cell_bank.values()):
            tmp_cell = top_1 + cell
            tmp_cell %= input.shape[1]
            tmp_cell = tmp_cell.unsqueeze(-1)
            metrix_thr = torch.gather(input.unsqueeze(-1).expand(-1, -1, tmp_cell.size(-1)), 1, tmp_cell)
            soft_max = torch.softmax(metrix_thr, dim=1)
            output[:, idx] = torch.einsum("bkm, bkm -> bm", search_space[tmp_cell], soft_max).squeeze()
        output = self.fc(output)
        output = self.relu(output)
        self.clip_weights_values()
        return output

    def clip_weights_values(self):
        self.fc.weight.data = torch.clip(self.fc.weight.data, 0.1, 1)
        self.fc.weight.data /= torch.sum(self.fc.weight.data)
