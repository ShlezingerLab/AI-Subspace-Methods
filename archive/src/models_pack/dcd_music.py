import os
from pathlib import Path
import torch
from torch import nn

from src.metrics import CartesianLoss, RMSPELoss, MusicSpectrumLoss
from src.models_pack.parent_model import ParentModel
from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel
from src.methods_pack.music import MUSIC


class DCDMUSIC(ParentModel):
    """
    The Deep-Cascadede-defferntiable MUSIC is a suggested solution for localization in near-field.
    It uses 2 SubspaceNet:
    The first, is SubspaceNet+Esprit/RootMusic/MUSIC(with Maskpeak), to get the angle from the input tensor.
    The second, uses the first to extract the angles, and then uses the angles to get the distance.
    """

    def __init__(self, system_model: SystemModel, tau: int, diff_method: tuple = ("esprit", "music_1d"),
                 regularization: str = None, variant: str = "small",
                 norm_layer: bool = True, batch_norm: bool = False, psd_epsilon: float = 1e-6,
                 load_angle_branch: bool = False, angle_extractor: SubspaceNet = None, load_range_branch: bool = False):
        super(DCDMUSIC, self).__init__(system_model)
        self.tau = tau
        self.regularization = regularization
        self.psd_epsilon = psd_epsilon
        self.norm_layer = norm_layer
        self.batch_norm = batch_norm
        self.variant = variant
        self.angle_branch, self.range_branch = None, None # Holders for the angle and range branches
        self.__init_angle_branch(load_angle_branch, diff_method[0]) if angle_extractor is None else angle_extractor
        self.__init_range_branch(load_range_branch, diff_method[1])
        self.train_mode = "position" # Default to position mode
        # self.update_train_mode("angle") # "angle", "range" or "position"
        self.train_loss, self.validation_loss = None, None
        self.__set_criterion()
        self.use_gt = True
        if self.use_gt:
            self.range_branch.diff_method.init_cells(0.05)

    def forward(self, x: torch.Tensor, number_of_sources: int = None, ground_truth_angles: torch.Tensor = None):
        if self.train_mode == "angle":
            angles, sources_estimation, eigen_regularization = self.angle_branch_forward(x, number_of_sources)
            return angles, sources_estimation, eigen_regularization
        elif self.train_mode == "range":
            with torch.no_grad():
                self.angle_branch.eval()
                angles, sources_estimation, _ = self.angle_branch_forward(x, number_of_sources)
            known_angles = ground_truth_angles if self.train_mode == "range" and ground_truth_angles is not None and self.use_gt else angles
            distances = self.range_branch_forward(x, number_of_sources=number_of_sources, known_angles=known_angles)
            return known_angles, distances, None, None
        elif self.train_mode == "position" or self.train_mode is None: # Default to position mode
            angles, sources_estimation, eigen_regularization = self.angle_branch_forward(x, number_of_sources)
            distances = self.range_branch_forward(x, number_of_sources, known_angles=angles)
            return angles, distances, sources_estimation, eigen_regularization

    def angle_branch_forward(self, x: torch.Tensor, number_of_sources: int = None):
        Rz = self.angle_branch.get_surrogate_covariance(x)
        angles_predictions, sources_estimation, eigen_regularization = self.angle_branch.diff_method(Rz, number_of_sources)
        return angles_predictions, sources_estimation, eigen_regularization

    def range_branch_forward(self, x: torch.Tensor, number_of_sources: int = None, known_angles: torch.Tensor = None):
        Rz = self.range_branch.get_surrogate_covariance(x)
        ranges_predictions = self.range_branch.diff_method(Rz, number_of_sources, known_angles)
        return ranges_predictions

    def training_step(self, batch, batch_idx):
        x, sources_num, labels = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"SubspaceNet.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        x = x.to(self.device)
        angles = angles.to(self.device)
        ranges = ranges.to(self.device)
        eigen_regularization, sources_estimation = None, None
        if self.train_mode == "angle":
            angles_pred, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles_pred=angles_pred, angles=angles)
        elif self.train_mode == "range":
            angles_pred, ranges_pred, sources_estimation, eigen_regularization = self(x, sources_num,
                                                                                      ground_truth_angles=angles)
            loss, loss_angles, loss_ranges = self.train_loss(angles_pred=angles_pred, angles=angles,
                                                             ranges_pred=ranges_pred, ranges=ranges)
        else: # self.train_mode == "position":
            angles_pred, ranges_pred, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)

        acc = self.source_estimation_accuracy(sources_num, sources_estimation)
        loss = self.get_regularized_loss(loss, eigen_regularization)

        return loss, acc, eigen_regularization


    def validation_step(self, batch, batch_idx, is_test: bool=False):
        x, sources_num, labels = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"SubspaceNet.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        x = x.to(self.device)
        angles = angles.to(self.device)
        ranges = ranges.to(self.device)
        if self.train_mode == "angle":
            angles_pred, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.validation_loss(angles_pred, angles)
        else:
            angles_pred, ranges_pred, sources_estimation, eigen_regularization = self(x, sources_num,
                                                                                      ground_truth_angles=angles if not is_test else None)
            loss = self.validation_loss(angles_pred=angles_pred, angles=angles,
                                        ranges_pred=ranges_pred, ranges=ranges)
        if isinstance(loss, tuple):
            loss = loss[0]
        acc = self.source_estimation_accuracy(sources_num, sources_estimation)

        if is_test and self.train_mode == "position":
            _, loss_angle, loss_range = self.test_loss(angles_pred=angles_pred, angles=angles,
                                                       ranges_pred=ranges_pred, ranges=ranges)
            return (loss, loss_angle, loss_range), acc

        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, is_test=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        angles, distances, sources_estimation, eigen_regularization = self.forward(x, self.system_model.params.M)
        return angles, distances, sources_estimation

    def init_model_train_params(self, init_eigenregularization_weight: float = 0.001, init_cell_size: float = 0.2):
        self.__init_angle_branch_train_params(init_eigenregularization_weight)
        self.__init_range_branch_train_params(init_cell_size)
        self.__set_criterion()

    def __init_angle_branch_train_params(self, init_eigenregularization_weight: float):
        self.angle_branch.set_eigenregularization_schedular(init_value=init_eigenregularization_weight)

    def __init_range_branch_train_params(self, init_cell_size: float):
        if isinstance(self.range_branch.diff_method, MUSIC) and isinstance(self.train_loss, RMSPELoss):
            self.range_branch.diff_method.init_cells(init_cell_size)

    def __init_angle_branch(self, load_state: bool, diff_method: str):
        self.angle_branch = SubspaceNet(tau=self.tau, diff_method=diff_method, train_loss_type="rmspe",
                            system_model=self.system_model, field_type="far", regularization=self.regularization,
                            variant=self.variant, norm_layer=self.norm_layer, batch_norm=self.batch_norm,
                            psd_epsilon=self.psd_epsilon)
        self.load_angle_branch(load_state)

    def __init_range_branch(self, load_state: bool, diff_method: str):
        self.range_branch = SubspaceNet(tau=self.tau, diff_method=diff_method, train_loss_type="rmspe",
                            system_model=self.system_model, field_type="near", regularization=None,
                            variant=self.variant, norm_layer=self.norm_layer, batch_norm=self.batch_norm,
                            psd_epsilon=self.psd_epsilon)
        self.load_range_branch(load_state)

    def load_angle_branch(self, load_state: bool):
        self.angle_branch = self.__load_branch(load_state, "angle")

    def load_range_branch(self, load_state: bool):
        self.range_branch = self.__load_branch(load_state, "range")

    def __load_branch(self, load_state: bool, branch: str):
        model = self.angle_branch if branch == "angle" else self.range_branch

        if load_state:
            path = os.path.join(Path(__file__).parent.parent.parent, "data", "weights", model._get_name(), "final_models",
                                model.get_model_file_name())
            try:
                model.load_state_dict(torch.load(path + ".pt", map_location=self.device, weights_only=True))
            except FileNotFoundError as e:
                raise FileNotFoundError(f"DCDMUSIC.__init_{branch}_branch: Model state not found in {path}")
            print(f"DCDMUSIC.__init_{branch}_branch: Model state loaded from {path}")
        return model


    def print_model_params(self):
        params = self.get_model_params()
        name = f"tau={params.get('tau')}_diff_methods={params.get('diff_methods')[0]}_{params.get('diff_methods')[1]}"
        if self.regularization is not None:
            name += f"_reg={self.regularization}"
        return name

    def get_model_params(self):
        if str(self.angle_branch.diff_method).startswith("music"):
            angle_extractor_diff_method = str(self.angle_branch.diff_method) + "_" + str(self.angle_branch.train_loss)
        else:
            angle_extractor_diff_method = str(self.angle_branch.diff_method)
        if str(self.range_branch.diff_method).startswith("music"):
            diff_method = str(self.range_branch.diff_method) + "_" + str(self.range_branch.train_loss)
        else:
            diff_method = str(self.range_branch.diff_method)
        return {"tau": self.tau, "diff_methods": (angle_extractor_diff_method ,diff_method)}



    def __set_criterion(self):
        if self.train_mode == "angle":
            self.train_loss = RMSPELoss(balance_factor=1.0)
            self.validation_loss = RMSPELoss(balance_factor=1.0)
        elif self.train_mode == "range":
            self.train_loss = RMSPELoss(balance_factor=0.0)
            self.validation_loss = RMSPELoss(balance_factor=0.0)
        elif self.train_mode == "position":
            self.train_loss = CartesianLoss()
            self.validation_loss = CartesianLoss()
            self.test_loss = RMSPELoss(balance_factor=1.0)
        else:
            raise ValueError(f"DCDMUSIC.__set_criterion: Unknown train mode {self.train_mode}")

    def update_criterion(self):
        self.__set_criterion()

    def update_train_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ["angle", "range", "position"]:
            raise ValueError(f"DCDMUSIC.update_train_mode: Unknown mode {mode}")
        self.train_mode = mode
        print(f"DCDMUSIC.update_train_mode: Train mode updated to {mode}")
        if mode == "angle":
            self.__set_branch_requires_grad(True, branch="angle")
            self.__set_branch_requires_grad(False, branch="range")
        elif mode == "range":
            self.__set_branch_requires_grad(False, branch="angle")
            self.__set_branch_requires_grad(True, branch="range")
        else:
            self.__set_branch_requires_grad(True, branch="angle")
            self.__set_branch_requires_grad(True, branch="range")
        self.update_criterion()

    def switch_train_mode(self):
        if self.train_mode == "angle":
            self.update_train_mode("range")
        elif self.train_mode == "range":
            self.update_train_mode("position")
        else:
            self.update_train_mode("angle")

    def __set_branch_requires_grad(self, requires_grad: bool, branch: str):
        if branch == "angle":
            for param in self.angle_branch.parameters():
                param.requires_grad = requires_grad
        elif branch == "range":
            for param in self.range_branch.parameters():
                param.requires_grad = requires_grad

    def _get_name(self):
        name = "DCDMUSIC"
        if self.variant == "V2":
            name += f"_V2"
        return name

    def train(self, T: bool = True):
        if self.train_mode == "angle":
            self.angle_branch.train(T)
            self.range_branch.eval()
        elif self.train_mode == "range":
            self.range_branch.train(T)
            self.angle_branch.eval()
        else:
            self.angle_branch.train(T)
            self.range_branch.train(T)

    def get_regularized_loss(self, loss, l_eig=None):
        if l_eig is not None:
            loss_r = loss + self.angle_branch.eigenregularization_weight * l_eig
        else:
            loss_r = loss
        return torch.sum(loss_r)

    def get_eigenregularization_weight(self):
        if self.train_mode in ["angle", "position"]:
            return self.angle_branch.eigenregularization_weight
        else:
            return None
