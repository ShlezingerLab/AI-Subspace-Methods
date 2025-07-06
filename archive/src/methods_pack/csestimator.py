"""
Implemntation of the method discribed in:
Near field source localization using sparse recovery techniques
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.system_model import SystemModel
from src.metrics import CartesianLoss, RMSPELoss
from src.config import device

from sklearn.linear_model import Lasso
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class CsEstimator(nn.Module):
    def __init__(self, system_model: SystemModel, solver_iter=5000, solver_mu=.01):
        super(CsEstimator, self).__init__()
        self.device = device
        self.system_model = system_model
        self.angles_dict = None
        self.ranges_dict = None
        self.steering_matrix_dict = None
        self.solver_iter = solver_iter
        self.solver_mu = solver_mu

        self.__init_grid_params()
        self.__init_steering_matrix_dict()
        self.__init_criteria()


    def forward(self, y):
        estimated_signal = self.l1_regularized_least_squares(y, "sklearn")
        estimated_doa, estimated_range, estimated_signal = self.estimate_doa_range(estimated_signal)

        return (estimated_doa.to(self.device), estimated_range.to(self.device)), estimated_signal

    def estimate_doa_range(self, estimated_signal, aggregation: str="sparse"):
        """
        Estimate the DOA and range of the sources from the estimated signal.
        """
        # Compute the l2 norm of the estimated signal
        estimated_signal_norm = torch.abs(estimated_signal)
        if aggregation == "sparse":
            # take the sparsest signal from the last dimension
            norms = torch.norm(estimated_signal_norm, p=1, dim=1)
            min_indices = torch.argmin(norms, dim=-1)
            estimated_signal_norm = estimated_signal_norm[:, :, min_indices].squeeze(-1)
        elif aggregation == "mean":
            estimated_signal_norm = torch.mean(estimated_signal_norm, dim=-1)

        if estimated_signal_norm.dim() == 2:
            max_indices = torch.zeros(estimated_signal_norm.size(0), self.system_model.params.M, device=self.device)
            for i in range(estimated_signal_norm.size(0)):
                flatten_signal = estimated_signal_norm[i].cpu().numpy()
                peaks = find_peaks(flatten_signal, distance=int(len(self.ranges_dict)*2))[0]
                sorted_peaks = peaks[np.argsort(flatten_signal[peaks])[::-1]]
                max_indices[i] = torch.tensor(sorted_peaks[:self.system_model.params.M], device=self.device)
            # max_indices = torch.topk(estimated_signal_norm, self.system_model.params.M, dim=1).indices
        elif estimated_signal_norm.size(-1) > self.system_model.params.M:
            estimated_signal_norm = torch.mean(estimated_signal_norm, dim=-1)
            max_indices = torch.topk(estimated_signal_norm, self.system_model.params.M, dim=1).indices
        else:
            max_indices = torch.argmax(estimated_signal_norm, dim=1)


        max_row = max_indices // self.steering_matrix_dict.size(2)
        max_col = max_indices % self.steering_matrix_dict.size(2)

        estimated_doa = self.angles_dict[max_row.to("cpu").to(int)]
        estimated_range = self.ranges_dict[max_col.to("cpu").to(int)]

        return estimated_doa, estimated_range, estimated_signal_norm

    def l1_regularized_least_squares(self, y, mode: str="torch"):
        """
        Solve the ℓ₁-regularized weighted least squares problem.
        """
        if mode == "torch":
            # Convert inputs to tensors if they aren't already
            A = torch.as_tensor(self.steering_matrix_dict.view(y.size(1), -1), dtype=torch.complex128, device=self.device)
            y = torch.as_tensor(y, dtype=torch.complex128, device=self.device)

            # Initialize x
            x = torch.randn(y.size(0), A.size(1), y.size(-1), dtype=torch.complex128, device=self.device)
            x.requires_grad_(True)  # Enable gradient tracking

            # Define optimizer
            optimizer = optim.Adam([x], lr=0.001)

            # Optimization loop
            for _ in range(self.solver_iter):
                def closure():
                    optimizer.zero_grad()
                    # Compute forward pass
                    loss = self.compute_cost(y, A, x)
                    loss.backward()
                    return loss

                optimizer.step(closure)

            return x.detach()
        elif mode == "sklearn":
            # sklearn doesn't support complex numbers and batchwise computation
            A = self.steering_matrix_dict.view(y.size(1), -1).cpu().numpy()
            A_real_imag = np.block([
                [A.real, -A.imag],
                [A.imag, A.real]
            ])
            res = []
            solver = Lasso(alpha=self.solver_mu, fit_intercept=False, max_iter=self.solver_iter)
            for batch_idx in range(y.size(0)):
                y_batch = y[batch_idx].cpu().numpy()
                tmp_res = []
                for t in range(y.shape[-1]):
                    y_tmp= np.concatenate((np.real(y_batch[:, t]), np.imag(y_batch[:, t])))
                    solver.fit(A_real_imag, y_tmp)
                    x_real_imag = solver.coef_
                    x = x_real_imag[:A.shape[1]] + 1j * x_real_imag[A.shape[1]:]
                    tmp_res.append(torch.from_numpy(x))
                tmp_res = torch.stack(tmp_res, dim=1)
                res.append(tmp_res)
            est_x = torch.stack(res, dim=0).to(self.device)
            return est_x

    def l1_svd_regularized_least_squares(self, y):
        # get a compact form of the input matrix
        u, s, vh = torch.linalg.svd(y)
        # get the compact NxS matrix
        d_k = torch.zeros(vh.shape[0], vh.shape[1], self.system_model.params.M, dtype=torch.complex128, device=self.device)
        d_k[:, :self.system_model.params.M, :self.system_model.params.M] = torch.eye(self.system_model.params.M, dtype=torch.complex128, device=self.device)
        # get the compact form of the input matrix
        y_sv = torch.bmm(torch.bmm(y, torch.conj(vh).transpose(1,2)), d_k)
        # get the l1 regularized least squares
        x = self.l1_regularized_least_squares(y_sv)
        return x


    def compute_cost(self, y, A, x):
        # Compute the l2 norm of the residual whereas: Y is B x N x T, A is NxS, X is BxSxt
        residual = y - torch.einsum("ns, bst -> bnt", A, x)
        residual_norm = torch.mean(torch.norm(residual, p=2, dim=1) ** 2, dim=1)
        l1_regularization = torch.norm(x, p=1, dim=1).mean(dim=-1).mean()
        l2_regularization = torch.norm(x, p=2, dim=1).mean(dim=-1).mean()

        loss = residual_norm + self.solver_mu * l1_regularization + self.solver_mu * l2_regularization
        return loss


    def test_step(self, batch, batch_idx: int, model: nn.Module=None):
        x, sources_num, label, masks = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        test_length = x.shape[0]
        x = x.to(self.device)
        if max(sources_num) * 2 == label.shape[1]:
            angles, ranges = torch.split(label, max(sources_num), dim=1)
            angles = angles.to(self.device)
            ranges = ranges.to(self.device)
            masks, _ = torch.split(masks, max(sources_num), dim=1)  # TODO
        else:
            angles = label.to(self.device)  # only angles

        # Check if the sources number is the same for all samples in the batch
        if (sources_num != sources_num[0]).any():
            # in this case, the sources number is not the same for all samples in the batch
            raise Exception(f"train_model:"
                            f" The sources number is not the same for all samples in the batch.")
        else:
            sources_num = sources_num[0]

        predictions, estimated_signal = self(x)
        if isinstance(predictions, tuple):
            angles_prediction, ranges_prediction = predictions
            rmspe = self.criterion(angles_prediction, angles, ranges_prediction, ranges).item()
            _, rmspe_angle, rmspe_range = self.separated_criterion(angles_prediction, angles, ranges_prediction, ranges)
            rmspe = (rmspe, rmspe_angle.item(), rmspe_range.item())
        else:
            rmspe = self.criterion(predictions, angles).item()

        return rmspe, 0, test_length

    def __init_grid_params(self):
        """
        Set values for the angle and range dict, depends on the scenario.
        Returns:
            None.

        """
        angle_range = np.deg2rad(self.system_model.params.doa_range)
        angle_resolution = np.deg2rad(self.system_model.params.doa_resolution) / 2
        # if it's the Far field case, need to init angles range.
        self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution, device="cpu",
                                        dtype=torch.float64).requires_grad_(False).to(torch.float64)
        if self.system_model.params.field_type.startswith("near"):
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            fraunhofer_ratio = self.system_model.params.max_range_ratio_to_limit
            distance_resolution = self.system_model.params.range_resolution / 2
            self.ranges_dict = torch.arange(np.ceil(fresnel),
                                            fraunhofer * fraunhofer_ratio + distance_resolution,
                                            distance_resolution,
                                            device="cpu", dtype=torch.float64).requires_grad_(False)

    def __init_steering_matrix_dict(self):
        self.steering_matrix_dict = self.system_model.steering_vec_near_field(self.angles_dict, self.ranges_dict,
                                                                              nominal=True, generate_search_grid=True)

    def __init_criteria(self):
        self.criterion = CartesianLoss()
        self.separated_criterion = RMSPELoss(1.0)

    def plot_signal(self, signal_norm, ground_truth=None, estimations=None):
        reshaped_signal = signal_norm.view(self.steering_matrix_dict.size(1), self.steering_matrix_dict.size(2)).cpu().detach().numpy()
        angle_labels = np.rad2deg(self.angles_dict.cpu().detach().numpy())
        range_labels = self.ranges_dict.cpu().detach().numpy()

        plt.figure()
        plt.imshow(reshaped_signal, aspect='auto',
                   extent=[range_labels[0], range_labels[-1], angle_labels[0], angle_labels[-1]], origin='lower')
        if ground_truth is not None:
            plt.scatter(ground_truth[1], np.rad2deg(ground_truth[0]), color='red', marker='x', label='Ground Truth')
        if estimations is not None:
            plt.scatter(estimations[1], np.rad2deg(estimations[0]), color='blue', marker='o', label='Estimations')
        plt.colorbar()
        plt.xlabel('Range')
        plt.ylabel('Angle')
        plt.title('Signal')
        plt.legend()
        plt.show()
