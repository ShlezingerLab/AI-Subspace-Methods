import torch
import torch.nn as nn

from src.metrics import RMSPELoss
from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel


class ESPRIT(SubspaceMethod):
    def __init__(self, system_model: SystemModel, model_order_estimation: str = None):
        super().__init__(system_model, model_order_estimation)
        self.__init_criteria()

    def forward(self, cov: torch.Tensor, sources_num: torch.tensor = None):
        if sources_num is None:
            M = self.system_model.params.M
        else:
            M = sources_num
        # get the signal subspace
        signal_subspace, _, sources_estimation, regularization = self.subspace_separation(
            cov,
            number_of_sources=M
        )
        # create 2 overlapping matrices
        upper = signal_subspace[:, :-1]
        lower = signal_subspace[:, 1:]
        phi = torch.linalg.lstsq(upper, lower)[0]  # identical to pinv(A) @ B but faster and stable.
        eigvalues, _ = torch.linalg.eig(phi)
        eigvals_phase = torch.angle(eigvalues)
        prediction = -1 * torch.arcsin((1 / torch.pi) * eigvals_phase)

        return prediction, sources_estimation, regularization

    def test_step(self, batch, batch_idx, model: nn.Module=None):
        x, sources_num, label = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        test_length = x.shape[0]
        x = x.to(self.device)
        if max(sources_num) * 2 == label.shape[1]:
            angles, _ = torch.split(label, max(sources_num), dim=1)
            angles = angles.to(self.device)
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
            if self.system_model.params.signal_nature == "coherent":
                # Spatial smoothing
                Rx = self.pre_processing(x, mode="sps")
            else:
                # Conventional
                Rx = self.pre_processing(x, mode="sample")
        angles_prediction, sources_num_estimation, _ = self(Rx, sources_num=sources_num)
        rmspe = self.criterion(angles_prediction, angles).sum().item()
        acc = self.source_estimation_accuracy(sources_num, sources_num_estimation)

        return rmspe, acc, test_length

    def __str__(self):
        return "esprit"

    def __init_criteria(self):
        self.criterion = RMSPELoss(balance_factor=1.0)

# LEGACY CODE

# def esprit(Rz: torch.Tensor, M: int, batch_size: int):
#     """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
#         MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
#         as it accepts the surrogate covariance matrix.
#         it is equivalent to src.methods: RootMUSIC.narrowband() method.
#
#     Args:
#     -----
#         Rz (torch.Tensor): Focused covariance matrix
#         M (int): Number of sources
#         batch_size: the number of batches
#
#     Returns:
#     --------
#         doa_batches (torch.Tensor): The predicted doa, over all batches.
#     """
#
#     doa_batches = []
#
#     Bs_Rz = Rz
#     for iter in range(batch_size):
#         R = Bs_Rz[iter]
#         # Extract eigenvalues and eigenvectors using EVD
#         eigenvalues, eigenvectors = torch.linalg.eig(R)
#
#         # Get signal subspace
#         Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
#         # Separate the signal subspace into 2 overlapping subspaces
#         Us_upper, Us_lower = (
#             Us[0: R.shape[0] - 1],
#             Us[1: R.shape[0]],
#         )
#         # Generate Phi matrix
#         phi = torch.linalg.pinv(Us_upper) @ Us_lower
#         # Find eigenvalues and eigenvectors (EVD) of Phi
#         phi_eigenvalues, _ = torch.linalg.eig(phi)
#         # Calculate the phase component of the roots
#         eigenvalues_angels = torch.angle(phi_eigenvalues)
#         # Calculate the DoA out of the phase component
#         doa_predictions = -1 * torch.arcsin((1 / torch.pi) * eigenvalues_angels)
#         doa_batches.append(doa_predictions)
#
#     return torch.stack(doa_batches, dim=0)
