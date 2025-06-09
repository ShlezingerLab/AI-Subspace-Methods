import torch.nn as nn
import torch
from itertools import permutations
from src.config import device

import numpy as np

BALANCE_FACTOR = 1.0

class RMSPELoss(nn.Module):
    """
    Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.
    In case of Near field, the arguments could be the predicted ranges and the target ranges in addition to the angles.
    The minimal rmse results is used over the angles and projected to the range.
    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """

    def __init__(self, balance_factor=None):
        super(RMSPELoss, self).__init__()
        self.device = device
        if balance_factor is None:
            self.balance_factor = nn.Parameter(torch.Tensor([BALANCE_FACTOR])).to(self.device).to(torch.float64)
        else:
            self.balance_factor = nn.Parameter(torch.Tensor([balance_factor])).to(self.device).to(torch.float64)

    def forward(self, angles_pred: torch.Tensor, angles: torch.Tensor,
                ranges_pred: torch.Tensor = None, ranges: torch.Tensor = None):
        """
        Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa,
        and possibly distance_predictions and distance.
        The predicted values and target values are expected to be in radians for the DOA values.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSE.
        The minimum RMSE value among all permutations is selected for each sample,
         including the RMSE for the distance values with the same permutation.
        Finally, the method averged the RMSE values for all samples in the batch and returns the result as the computed loss.

        Args:
            angles_pred (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            angles (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            ranges_pred (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            ranges (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        # Calculate RMSPE loss for only DOA
        num_sources = angles_pred.shape[1]
        perm = list(permutations(range(num_sources), num_sources))
        num_of_perm = len(perm)

        err_angle = (angles_pred[:, perm] - torch.tile(angles[:, None, :], (1, num_of_perm, 1)).to(torch.float32))
        # Calculate error with modulo pi in the range [-pi/2, pi/2]
        err_angle += torch.pi / 2
        err_angle %= torch.pi
        err_angle -= torch.pi / 2
        rmspe_angle_all_permutations = np.sqrt(1 / num_sources) * torch.linalg.norm(err_angle, dim=-1)
        if ranges is None:
            rmspe, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
        else:
            rmspe_angle, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
            # create the projected permutation using the min_idx
            projected_permutations = torch.tensor(perm, dtype=torch.long, device=self.device)[min_idx]
            projected_ranges_pred = torch.gather(ranges_pred, 1, projected_permutations)
            projected_err_ranges = projected_ranges_pred - ranges
            projected_rmse_ranges = np.sqrt(1 / num_sources) * torch.linalg.norm(projected_err_ranges, dim=-1)


            rmspe = self.balance_factor * rmspe_angle + (1 - self.balance_factor) * projected_rmse_ranges
        # result = torch.sum(rmspe)
        if ranges is None:
            return rmspe
        else:
            # result_angle = torch.sum(rmspe_angle)
            # result_distance = torch.sum(projected_rmse_ranges)
            return rmspe, rmspe_angle, projected_rmse_ranges

    def adjust_balance_factor(self, loss=None):
        self.balance_factor = 0.1

    def __str__(self):
        return "RMSPELoss"

    def __repr__(self):
        return "RMSPELoss(balance_factor={})".format(self.balance_factor.item())