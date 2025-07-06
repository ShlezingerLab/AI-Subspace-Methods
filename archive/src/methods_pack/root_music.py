import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel
from src.metrics import RMSPELoss


class RootMusic(SubspaceMethod):
    def __init__(self, system_model: SystemModel, model_order_estimation: str = "threshold"):
        super(RootMusic, self).__init__(system_model, model_order_estimation)
        self.__init_criteria()
        self.all_predictions_angles = None
        self.all_predictions_roots = None

    def forward(self, cov: torch.Tensor, sources_num: torch.tensor = None):
        batch_size = cov.shape[0]
        _, noise_subspace, source_estimation, _ = self.subspace_separation(cov, number_of_sources=sources_num)
        poly_generator = torch.bmm(noise_subspace, noise_subspace.conj().transpose(1, 2))
        diag_sum = self.sum_of_diag(poly_generator)
        roots = self.find_roots(diag_sum)
        self.all_predictions_angles = self.get_doa_from_roots(roots)
        self.all_predictions_roots = roots
        # self.plot_spectrum(batch_idx=0)
        # the actual prediction is for roots that are closest to the unit circle.
        roots = self.extract_roots_closest_unit_circle(roots, k=sources_num)
        angles_prediction = self.get_doa_from_roots(roots)
        return angles_prediction, source_estimation

    def get_doa_from_roots(self, roots):
        roots_phase = torch.angle(roots)
        angle_predicted = torch.arcsin(
            (1 / (2 * np.pi * self.system_model.dist_array_elems["narrowband"]) * self.system_model.params.wavelength) * roots_phase)
        return angle_predicted

    def extract_roots_closest_unit_circle(self, roots, k: int):
        distances = torch.abs(torch.abs(roots) - 1)
        sorted_indcies = torch.argsort(distances, dim=1)
        roots_sorted = roots.gather(1, sorted_indcies)
        # consider only the roots inside the unit circle
        closest_roots = torch.zeros(roots.shape[0], k).to(torch.complex128)
        for i in range(roots_sorted.shape[0]):
            filter_roots = roots_sorted[i, torch.abs(roots_sorted[i]) < 1]
            closest_roots[i] = filter_roots[:k]
        # get the k closest roots
        return closest_roots

    def sum_of_diag(self, tensor: torch.Tensor):
        tensor = self.__check_diag_sums_dim(tensor)
        N = tensor.shape[-1]
        diag_indcies = torch.linspace(-N + 1, N - 1, 2 * N - 1)
        sum_of_diags = torch.zeros(tensor.shape[0], 2 * N - 1).to(torch.complex128)
        for idx, diag_idx in enumerate(diag_indcies):
            sum_of_diags[:, idx] = torch.sum(torch.diagonal(tensor, dim1=1, dim2=2, offset=diag_idx.to(int)), dim=-1)
        return sum_of_diags

    def find_roots(self, coeffs: torch.Tensor):
        A = torch.diag(torch.ones(coeffs.shape[-1] - 2, dtype=coeffs.dtype), -1)
        A = A.repeat(coeffs.shape[0], 1, 1)  # repeat for all elements in the batch
        A[:, 0, :] = -torch.div(coeffs[:, 1:], coeffs[:, 0][:, None])
        roots = torch.linalg.eigvals(A)
        return roots

    def test_step(self, batch, batch_idx):
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

        if self.system_model.params.signal_nature == "coherent":
            # Spatial smoothing
            Rx = self.pre_processing(x, mode="sps")
        else:
            # Conventional
            Rx = self.pre_processing(x, mode="sample")
        angles_prediction, sources_num_estimation = self(Rx, sources_num=sources_num)
        rmspe = self.criterion(angles_prediction, angles).sum().item()
        acc = self.source_estimation_accuracy(sources_num, sources_num_estimation)

        return rmspe, acc, test_length

    def __init_criteria(self):
        self.criterion = RMSPELoss(balance_factor=1.0)

    def __check_diag_sums_dim(self, tensor):
        if len(tensor.shape) != 3:
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError("sum_of_diag: Input tensor shape should be 2 or 3 dim")
        else:
            if tensor.shape[-1] != tensor.shape[-2]:
                raise ValueError("sum_of_diag: input tensor should be square matrices as a batch.")
        return tensor

    def plot_spectrum(self, batch_idx: int = 0, ground_truth: list = None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, polar=True)
        # Set axis location and limits
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(90)
        ax.set_thetamax(-90)
        # plot roots ang angles
        for i, angle in enumerate(self.all_predictions_angles[batch_idx]):
            r = np.abs(self.all_predictions_roots[batch_idx][i])
            ax.set_ylim([0, 1.2])
            ax.set_yticks([0, 1])
            ax.plot([0, angle], [0, r], marker='o')
        if ground_truth is not None:
            for angle in ground_truth:
                ax.plot([angle], [1], marker='o', color='r')
        ax.set_xlabel("Angels [deg]")
        ax.set_ylabel("Amplitude")
        plt.title("Root-MUSIC Spectrum")
        plt.tight_layout()
        plt.show()


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    )
