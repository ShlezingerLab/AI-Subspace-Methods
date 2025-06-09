import torch
import torch.nn as nn
from itertools import permutations
from src.config import device

class CartesianLoss(nn.Module):
    def __init__(self):
        """
        Cartesian Loss for comparing predicted and true coordinates in Cartesian space.
        This loss function computes the Cartesian distance between predicted and true coordinates, assuming permutation invariance.
        """
        super(CartesianLoss, self).__init__()
        self.device = device

    def forward(self, angles_pred: torch.Tensor, angles: torch.Tensor, ranges_pred: torch.Tensor,
                ranges: torch.Tensor):
        """
        the input given is expected to contain angels and distances.
        """
        M = angles.shape[1]
        if angles_pred.shape[1] > angles.shape[1]:
            # in this case, randomly drop some of the predictions
            indices = torch.randperm(angles_pred.shape[1])[:M].to(self.device)
            angles_pred = torch.gather(angles_pred, 1, indices[None, :])
            ranges_pred = torch.gather(ranges_pred, 1, indices[None, :])

        elif angles_pred.shape[1] < angles.shape[1]:
            # add a random angle to the predictions
            random_angles = torch.distributions.uniform.Uniform(-torch.pi / 3, torch.pi / 3).sample(
                [angles_pred.shape[0], M - angles_pred.shape[1]])
            random_ranges = torch.distributions.uniform.Uniform(torch.min(ranges).item(),
                                                                torch.max(ranges).item()).sample(
                [angles_pred.shape[0], M - angles_pred.shape[1]])
            angles_pred = torch.cat((angles_pred, random_angles.to(self.device)), dim=1)
            ranges_pred = torch.cat((ranges_pred, random_ranges.to(self.device)), dim=1)

        number_of_samples = angles_pred.shape[0]
        true_x = torch.cos(angles) * ranges
        true_y = torch.sin(angles) * ranges
        coords_true = torch.stack((true_x, true_y), dim=2)
        pred_x = torch.cos(angles_pred) * ranges_pred
        pred_y = torch.sin(angles_pred) * ranges_pred
        coords_pred = torch.stack((pred_x, pred_y), dim=2)
        # need to consider all possible permutations for M sources
        perm = list(permutations(range(M), M))
        perm = torch.tensor(perm, dtype=torch.int64).to(self.device)
        num_of_perm = len(perm)

        error = torch.tile(coords_true[:, None, :, :], (1, num_of_perm, 1, 1)) - coords_pred[:, perm]
        cartesian_distance_all_permutations = torch.sqrt(torch.sum(error ** 2, dim=-1))
        mean_cartesian_distance_all_permutations = torch.mean(cartesian_distance_all_permutations, dim=-1)
        mean_cartesian_distance = torch.min(mean_cartesian_distance_all_permutations, dim=-1)
        return mean_cartesian_distance[0]

    def __str__(self):
        return "Cartesian RMSPE Loss"

    def __repr__(self):
        return "CartesianLoss()"