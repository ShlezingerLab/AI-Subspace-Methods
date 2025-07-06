import torch
import torch.nn as nn
import tqdm
import itertools
from itertools import product

from src.system_model import SystemModel
from src.utils import *


class MLE(nn.Module):
    """
    This is the Maximum Likelihood Estimation method for localization in Far and Near field environments.
    """

    def __init__(self, system_model: SystemModel):
        super(MLE, self).__init__()
        self.system_model = system_model
        self.angles = None
        self.distances = None
        self.subsets_angles = None
        self.subsets_distances = None
        self.__define_grid_params()
        self.search_grid = None
        self.__define_search_grid()
        self.projection_matrix_dict = {}
        self.__init_projection_matrix_dict()

    def forward(self, cov):
        """
        Perform the Maximum Likelihood Estimation method for localization in Far and Near field environments.

        Args:
            cov: the covariance tensor to preform the MLE one. size: BatchSizeXSensorsX Sensors
        Returns:
            the predicted angles and distances.
        """
        if self.system_model.params.field_type.startswith("Far"):
            return self.__far_field(cov)
        elif self.system_model.params.field_type.startswith("Near"):
            return self.__near_field(cov)

    def __far_field(self, cov):
        pass

    def __near_field(self, cov):
        """
        Perform the Maximum Likelihood Estimation method for localization in Near field environments.
        """
        # Currently assumes that the number of sources is known.
        number_of_sources = self.system_model.params.M

        # need to check all subsets of size number_of_sources in the search grid.
        # for each subset, calculate the likelihood function by the formula:
        # f(angles, distances) = Tr[P_A(angles, distances) * cov]]
        # where P_A(angles, distances) is the projection matrix of the given steering matrix given by psseudo inverse.

        angles_output = torch.zeros(cov.shape[0], number_of_sources, device=device, dtype=torch.float32,
                                    requires_grad=False)
        distances_output = torch.zeros(cov.shape[0], number_of_sources, device=device, dtype=torch.float32,
                                       requires_grad=False)
        likelihoods = torch.zeros(cov.shape[0], 1, device=device, dtype=torch.float32, requires_grad=False)

        # for idx_a, subset_angles in enumerate(tqdm(self.subsets_angles)):
        #     for idx_d, subset_distances in enumerate(self.subsets_distances):
        #         # calculate the projection matrix
        #         steering_matrix = self.search_grid[:, subset_angles, subset_distances] # size: sensorsXsources
        #         projection_matrix = self.projection_matrix_dict.get((subset_angles, subset_distances))
        #         if projection_matrix is None:
        #             projection_matrix = steering_matrix @ torch.pinverse(steering_matrix)
        #             self.projection_matrix_dict[(subset_angles, subset_distances)] = projection_matrix
        for (subset_angles, subset_distances), projection_matrix in tqdm(self.projection_matrix_dict.items()):
            # calculate the likelihood function
            likelihood_tmp = self.__calc_liklihood(cov, projection_matrix)  # size: batch_sizeX1
            mask = (likelihoods < likelihood_tmp).repeat(1, number_of_sources)

            # Ensure subset_angles and subset_distances are tensors and expand dimensions if necessary
            subset_angles = torch.tensor(subset_angles, device=device)
            subset_distances = torch.tensor(subset_distances, device=device)

            # Expand dimensions to match the batch size
            expanded_angles = self.angles[subset_angles].unsqueeze(0).expand(cov.shape[0], -1)
            expanded_distances = self.distances[subset_distances].unsqueeze(0).expand(cov.shape[0], -1)

            # Apply the mask to update angles_output and distances_output
            angles_output[mask] = expanded_angles[mask]
            distances_output[mask] = expanded_distances[mask]

            # Update likelihoods with the new values
            likelihoods[likelihoods < likelihood_tmp] = likelihood_tmp[likelihoods < likelihood_tmp]
        return torch.Tensor(angles_output), torch.Tensor(distances_output)

    def __calc_liklihood(self, cov, projection_matrix):
        if cov.dim() == 2:
            return torch.abs(torch.trace(projection_matrix @ cov))
        else:
            return torch.stack([torch.abs(torch.trace(projection_matrix @ c)) for c in cov])[:, None].to(device).to(
                torch.float32)

    def __define_grid_params(self):
        M = self.system_model.params.M
        if self.system_model.params.field_type.startswith("Far"):
            self.angles = torch.linspace(-np.pi / 2, np.pi / 2, 361)
            self.subsets_angles = list(itertools.combinations(range(self.angles.shape[0]), M))
        elif self.system_model.params.field_type.startswith("Near"):
            self.angles = torch.arange(-np.pi / 6, np.pi / 6, torch.pi / 90, device=device, dtype=torch.float32,
                                       requires_grad=False)
            fresnel = float(np.floor(self.system_model.fresnel))
            fraunhofer = float(np.round(self.system_model.fraunhofer, decimals=1) + 1)
            self.distances = torch.arange(fresnel, fraunhofer // 2, step=1,
                                          device=device, dtype=torch.float32, requires_grad=False)
            self.subsets_angles = list(itertools.combinations(range(self.angles.shape[0]), M))
            self.subsets_distances = list(itertools.combinations(range(self.distances.shape[0]), M))

    def __init_projection_matrix_dict(self):
        # Flatten the list of subsets and create a tensor for all combinations
        all_combinations = list(product(self.subsets_angles, self.subsets_distances))
        subset_angles, subset_distances = zip(*all_combinations)

        # Convert to tensors
        subset_angles_tensor = torch.tensor(subset_angles, device=device)
        subset_distances_tensor = torch.tensor(subset_distances, device=device)

        # Calculate the steering matrices for all combinations
        steering_matrices = self.search_grid[:, subset_angles_tensor,
                            subset_distances_tensor]  # Shape: sensors x combinations x sources

        # Compute the pseudo-inverse for each steering matrix and store in the dictionary
        projection_matrix_dict = {}
        for i, (angles, distances) in enumerate(tqdm(all_combinations)):
            steering_matrix = steering_matrices[:, i, :].to(device)
            projection_matrix = steering_matrix @ torch.linalg.pinv(steering_matrix)
            projection_matrix_dict[(tuple(angles), tuple(distances))] = projection_matrix

        self.projection_matrix_dict = projection_matrix_dict

    def __define_search_grid(self):
        if self.system_model.params.field_type.startswith("Far"):
            self.__define_search_grid_far()
        elif self.system_model.params.field_type.startswith("Near"):
            self.__define_search_grid_near()

    def __define_search_grid_near(self):
        dist_array_elems = self.system_model.dist_array_elems["NarrowBand"]
        angels = self.angles[:, None].to(device)
        distances = self.distances[:, None].to(device)
        #
        array = torch.Tensor(self.system_model.array[:, None]).to(torch.float32).to(device)
        array_square = torch.pow(array, 2).to(torch.float64)

        first_order = torch.einsum("nm, na -> na",
                                   array,
                                   torch.sin(angels).repeat(1, self.system_model.params.N).T * dist_array_elems)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(angels) * dist_array_elems, 2), distances.T)
        second_order = second_order[:, :, None].repeat(1, 1, self.system_model.params.N)
        second_order = torch.einsum("nm, nda -> nda",
                                    array_square,
                                    torch.transpose(second_order, 2, 0)).transpose(1, 2)

        first_order = first_order[:, :, None].repeat(1, 1, second_order.shape[-1])

        time_delay = first_order + second_order

        steering_matrix = torch.exp(2 * -1j * torch.pi * time_delay)
        self.search_grid = steering_matrix

    def __define_search_grid_far(self):
        pass
