"""
SubspaceNet: model-based deep learning algorithm as described in:
        "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
"""
import torch
import torch.nn as nn

from src.metrics import RMSPELoss, CartesianLoss, MusicSpectrumLoss
from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.utils import gram_diagonal_overload, validate_constant_sources_number, L2NormLayer, AntiRectifier, TraceNorm, LearnableSkipConnection, MultipleInputIdentity

from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.root_music import RootMusic, root_music
from src.methods_pack.beamformer import Beamformer


class SubspaceNet(ParentModel):
    def __init__(self, tau: int, diff_method: str = "root_music", train_loss_type: str="rmspe",
                 system_model: SystemModel = None, field_type: str = "far", regularization: str = None, variant: str = "small",
                  norm_layer: bool=True, psd_epsilon: float=.01, batch_norm: bool=False, skip_connection: bool=False,
                  use_eigenregularization: bool = True, initialize_eigenregularization_weight: float = 1e-3):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            diff_method (str): Differentiable subspace method.
            train_loss_type (str): Training loss type.
            system_model (SystemModel): System model.
            field_type (str): Field type.
            regularization (str): Regularization method.
            variant (str): Model variant.
            norm_layer (bool): Normalization layer.
            psd_epsilon (float): PSD epsilon.
            batch_norm (bool): Batch normalization.

        """
        super(SubspaceNet, self).__init__(system_model)
        # set model parameters
        self.field_type = field_type.lower()
        self.tau = tau
        self.diff_method = None # Holder for the differentiable subspace method
        # set model architecture
        self.p = 0.2
        self.regularization = regularization
        self.psd_epsilon = psd_epsilon
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.extra_conv4 = nn.Identity() # initialize as identity, set up in the __setup_model_variant
        self.extra_deconv1 = nn.Identity() # initialize as identity, set up in the __setup_model_variant
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.drop_out = nn.Dropout(self.p)
        self.antirectifier = AntiRectifier()
        self.skip_connection = MultipleInputIdentity() # initialize as identity, set up in the __setup_skip_connection
        
        self.__setup_model_variant(variant)
        self.__setup_norm_layer(norm_layer)
        self.__setup_batch_norm(batch_norm)
        self.__setup_skip_connection(skip_connection)

        # set model training parameters
        self.train_loss, self.validation_loss, self.test_loss, self.test_loss_separated = None, None, None, None
        self.__set_criterion(train_loss_type)
        self.__set_diff_method(diff_method, system_model)
        if use_eigenregularization:
            self.set_eigenregularization_schedular(init_value=initialize_eigenregularization_weight)

    def get_surrogate_covariance(self, x: torch.Tensor):
        """
        This function is the "real" forward pass of the SubspaceNet.
        It receives the input tensor and returns the surrogate covariance matrix.
        Args:
            x: the input tensor of shape [Batch size, N, T]

        Returns:
            Rz: the surrogate covariance matrix of shape [Batch size, N, N]
        """
        x0 = self.pre_processing(x)
        # Rx_tau shape: [Batch size, tau, 2N, N]
        # N = x.shape[-1]
        _, _, _, N = x0.shape
        empirical_cov = torch.complex(x0[:, 0,:N], x0[:, 0, N:])
        ############################
        ## Architecture flow ##
        # CNN block #1
        x1 = self.conv1(x0) # Shape: [Batch size, 16, 2N-1, N-1]
        x = self.antirectifier(x1) # Shape: [Batch size, 32, 2N-1, N-1]
        # CNN block #2
        x2 = self.conv2(x) # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x2) # Shape: [Batch size, 64, 2N-2, N-2]
        # CNN block #3
        x = self.conv3(x) # Shape: [Batch size, 64, 2N-3, N-3]
        x = self.antirectifier(x) # Shape: [Batch size, 128, 2N-3, N-3]

        # Additional CNN block for the big variant
        x = self.extra_conv4(x) # Shape: [Batch size, 128, 2N-4, N-4]
        # Additional Deconv block for the big variant
        x = self.extra_deconv1(x) # Shape: [Batch size, 64, 2N-3, N-3]

        x = self.deconv2(x) # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x + x2 if self.variant == "V2" else x) # Shape: [Batch size, 64, 2N-2, N-2]
        # DCNN block #3
        x = self.deconv3(x)     # Shape: [Batch size, 16, 2N-1, N-1]
        x = self.antirectifier(x + x1 if self.variant == "V2" else x) # Shape: [Batch size, 32, 2N-1, N-1]
        # DCNN block #4
        x = self.drop_out(x)
        Rx = self.deconv4(x)  # Shape: [Batch size, 1, 2N, N]  + x0[:, 0].unsqueeze(1)

        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag).to(torch.complex128)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        eps = self.__adjust_psd_eps()
        Kx_tag = self.norm_layer(Kx_tag)
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=eps) # Shape: [Batch size, N, N]
        Rz = self.skip_connection(Rz, empirical_cov) # Shape: [Batch size, N, N]
        # Rz = self.norm_layer(Rz)
        return Rz

    def forward(self, x: torch.Tensor, sources_num: torch.tensor = None):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            x (torch.Tensor): Input tensor of shape [Batch size, N, T].
            sources_num (torch.Tensor): The number of sources in the signal.
            known_angles (torch.Tensor): The known angles for the near-field scenario.

        Returns:
        --------
            The output of the forward pass.
        ------
        """
        Rz = self.get_surrogate_covariance(x)

        if self.field_type.startswith("far"):
            return self.__forward_far_field(Rz, sources_num)

        elif self.field_type == "near":
            return self.__forward_near_field(Rz, sources_num)

    def __forward_far_field(self, cov: torch.Tensor, sources_num: torch.tensor = None):
        """
        Forward pass for the far-field scenario.

        Args:
        -----
            cov (torch.Tensor): The surrogate covariance matrix.
            sources_num (torch.Tensor): The number of sources in the signal.

        Returns:
        --------
            noise_subspace (torch.Tensor): The noise subspace.
            source_estimation (torch.Tensor): The sources estimation.
            eigen_regularization (torch.Tensor): The eigen regularization.
            angles_pred (torch.Tensor): The predicted angles.
        --------
        """
        if isinstance(self.train_loss, MusicSpectrumLoss) and self.training:
            _, noise_subspace, source_estimation, eigen_regularization = self.diff_method.subspace_separation(cov,
                                                                                                              sources_num)
            return noise_subspace, source_estimation, eigen_regularization
        else:
            method_output = self.diff_method(cov, sources_num)
            if isinstance(self.diff_method, (RootMusic, ESPRIT, MUSIC)):
                pred_angles, sources_estimation, eigen_regularization = method_output
                return pred_angles.to(self.device), sources_estimation, eigen_regularization
            else:
                raise Exception(f"SubspaceNet.forward: Method {self.diff_method} is not defined for SubspaceNet")

    def __forward_near_field(self, cov: torch.Tensor, sources_num: torch.tensor = None):
        """
        Forward pass for the near-field scenario.
        Args:
            cov (torch.Tensor): The surrogate covariance matrix.
            sources_num (torch.Tensor): The number of sources in the signal.

        Returns:
            angles_pred (torch.Tensor): The predicted angles.
            distance_pred (torch.Tensor): The predicted distances.
            sources_estimation (torch.Tensor): The sources estimation.
            eigen_regularization (torch.Tensor): The eigen regularization.

        """
        if self.training and isinstance(self.train_loss, MusicSpectrumLoss):
            _, noise_subspace, source_estimation, eigen_regularization = self.diff_method.subspace_separation(cov,
                                                                                                              sources_num)
            return noise_subspace, source_estimation, eigen_regularization

        if isinstance(self.diff_method, MUSIC):
            predictions, sources_estimation, eigen_regularization = self.diff_method(
                cov, number_of_sources=sources_num)
            pred_angles, pred_ranges = predictions
            return pred_angles, pred_ranges, sources_estimation, eigen_regularization
        elif isinstance(self.diff_method, Beamformer):
            predictions = self.diff_method(
                cov, sources_num=sources_num)
            pred_angles, pred_ranges = predictions
            return pred_angles, pred_ranges, None, None
        else:
            raise Exception(f"SubspaceNet.forward: Method {self.diff_method} is not defined for SubspaceNet")

    def pre_processing(self, x):
        """
        The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be real tensors
        of size [batch, tau, 2N, N].

        Args:
        -----
            x (torch.Tensor): The complex input tensor of size [batch, N, T].

        Returns:
        --------
            Rx_tau (torch.Tensor): The pre-processed real tensor of size [batch, tau, 2N, N].
        """
        batch_size, N, T = x.shape
        Rx_tau = torch.zeros(batch_size, self.tau, 2 * N, N, device=self.device)
        meu = torch.mean(x, dim=-1, keepdim=True).to(self.device)
        center_x = x - meu
        if center_x.dim() == 2:
            center_x = center_x[None, :, :]

        for i in range(self.tau):
            x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
            x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
            Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
            Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim=1)
            Rx_tau[:, i, :, :] = Rx_lag

        return Rx_tau

    def __setup_skip_connection(self, skip_connection: bool):
        if skip_connection:
            self.skip_connection = LearnableSkipConnection(alpha=1e-1 if self.system_model.params.signal_nature == "non-coherent" else 1e-4)

    def __setup_model_variant(self, variant: str):
        if variant in ["big", "V2"]:
            self.extra_conv4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=2),
                AntiRectifier(),
                nn.Dropout(self.p))
            self.extra_deconv1 = nn.Sequential(
                nn.ConvTranspose2d(256, 64, kernel_size=2),
                AntiRectifier())
            self.variant = "V2"
        else:
            self.variant = ""

    def __setup_norm_layer(self, norm_layer: bool):
        if norm_layer:
            self.norm_layer = L2NormLayer()
            # self.norm_layer = TraceNorm()
        else:
            self.norm_layer = nn.Identity()

    def __setup_batch_norm(self, batch_norm: bool):
        # update conv and deconv layer to be sequential with batch norm
        if batch_norm:
            self.conv1 = nn.Sequential(
                self.conv1,
                nn.BatchNorm2d(16))
            self.conv2 = nn.Sequential(
                self.conv2,
                nn.BatchNorm2d(32))
            self.conv3 = nn.Sequential(
                self.conv3,
                nn.BatchNorm2d(64))
            self.deconv2 = nn.Sequential(
                self.deconv2,
                nn.BatchNorm2d(32))
            self.deconv3 = nn.Sequential(
                self.deconv3,
                nn.BatchNorm2d(16))
            self.deconv4 = nn.Sequential(
                self.deconv4,
                nn.BatchNorm2d(1))

    # Tested and not used
    def __init_reshaper(self):
        N = self.system_model.params.N
        h_dim = 2 * (N - self.reshaper_target_size) + 1
        w_dim = N - self.reshaper_target_size + 1
        if N > self.reshaper_target_size:
            conv = nn.Conv2d(in_channels=self.tau,
                             out_channels=self.tau,
                             kernel_size=(h_dim, w_dim),
                             stride=(1, 1), padding=0).to(self.device)
            self.reshaper = nn.Sequential(conv, nn.ReLU())
            # count the number of parameters
            print(f"Number of parameters in the reshaper: {sum(p.numel() for p in self.reshaper.parameters())}")
            print(f"Number of parameters in the model: {sum(p.numel() for p in self.parameters())}")
            # update the number of sensors of the diff method
            self.diff_method.update_number_of_sensors(self.reshaper_target_size)
        else:
            self.reshaper = nn.Identity()
        return self.reshaper

    def adjust_diff_method_temperature(self, epoch):
        if isinstance(self.diff_method, MUSIC) and isinstance(self.train_loss, (RMSPELoss, CartesianLoss)):
            if epoch % 20 == 0 and epoch != 0:
                self.diff_method.adjust_cell_size()
                print(f"Model temepartue updated --> {self.get_diff_method_temperature()}")

    def get_diff_method_temperature(self):
        if isinstance(self.diff_method, MUSIC):
            if self.diff_method.estimation_params in ["angle", "range"]:
                return self.diff_method.cell_size
            elif self.diff_method.estimation_params == "angle, range":
                return {"angle_cell_size": self.diff_method.cell_size_angle,
                        "distance_cell_size": self.diff_method.cell_size_range}

    def _get_name(self):
        if self.field_type == "far":
            name = super(SubspaceNet, self)._get_name()
        elif self.field_type == "near" and isinstance(self.train_loss, MusicSpectrumLoss):
            name = "NF" + super(SubspaceNet, self)._get_name()
        if self.variant:
            name += f"_{self.variant}"
        return name

    def print_model_params(self):
        params = self.get_model_params()

        repr = f"tau={params['tau']}_diff_method={params['diff_method']}_field_type={params['field_type']}_train_loss_type={params['train_loss_type']}"
        regularization = params.get("regularization", None)
        if regularization is not None:
            repr += f"_{regularization}"

        return repr

    def get_model_params(self):
        return {"tau": self.tau, "diff_method": str(self.diff_method), "field_type": self.field_type.lower(),
                "train_loss_type": str(self.train_loss), "regularization": self.regularization, "variant": self.variant}

    def init_model_train_params(self, init_eigenregularization_weight: float = 50):
        self.__set_criterion()
        self.set_eigenregularization_schedular(init_value=init_eigenregularization_weight)
        if isinstance(self.diff_method, MUSIC) and self.train_loss_type == "rmspe":
            self.diff_method.init_cells(0.1)

    def training_step(self, batch, batch_idx):
        if self.field_type == "far":
            return self.__training_step_far_field(batch, batch_idx)
        elif self.field_type.lower() == "near":
            return self.__training_step_near_field(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        if self.field_type == "far":
            return self.__validation_step_far_field(batch, batch_idx)
        elif self.field_type == "near":
            return self.__validation_step_near_field(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.field_type == "far":
            return self.__test_step_far_field(batch, batch_idx)
        elif self.field_type == "near":
            return self.__test_step_near_field(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if self.field_type == "far":
            return self.__prediction_step_far_field(batch, batch_idx)
        elif self.field_type == "near":
            return self.__prediction_step_near_field(batch, batch_idx)

    def __training_step_far_field(self, batch, batch_idx):
        x, sources_num, angles = self.__prepare_batch_far_field(batch)
        # if self.field_type != self.system_model.params.field_type:
        #     angles, _ = torch.split(angles, sources_num, dim=1)
        if isinstance(self.train_loss, MusicSpectrumLoss):
            noise_subspace, source_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(noise_subspace=noise_subspace, angles=angles)
        else:
            angles_pred, source_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles_pred=angles_pred, angles=angles)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        loss = self.get_regularized_loss(loss, eigen_regularization)
        return loss, acc, eigen_regularization

    def __validation_step_far_field(self, batch, batch_idx):
        x, sources_num, angles = self.__prepare_batch_far_field(batch)
        # if self.field_type != self.system_model.params.field_type:
        #     angles, _ = torch.split(angles, sources_num, dim=1)
        angles_pred, source_estimation, eigen_regularization = self(x, sources_num)
        loss = self.validation_loss(angles_pred=angles_pred, angles=angles)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc

    def __test_step_far_field(self, batch, batch_idx):
        return self.__validation_step_far_field(batch, batch_idx)

    def __prepare_batch_far_field(self, batch):
        x, sources_num, angles = batch
        validate_constant_sources_number(sources_num)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        angles = angles.to(self.device)
        sources_num = sources_num[0]
        return x, sources_num, angles

    def __prediction_step_far_field(self, batch, batch_idx):
        x = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        angles_pred, source_estimation, _ = self(x)
        return angles_pred, source_estimation

    def __training_step_near_field(self, batch, batch_idx):
        x, sources_num, angles, ranges = self.__prepare_batch_near_field(batch)
        if isinstance(self.train_loss, MusicSpectrumLoss):
            noise_subspace, source_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(noise_subspace=noise_subspace, angles=angles, ranges=ranges)
        else:
            angles_pred, distance_pred, source_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles_pred=angles_pred, angles=angles, ranges_pred=distance_pred, ranges=ranges)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        loss = self.get_regularized_loss(loss, eigen_regularization)
        return loss, acc, eigen_regularization

    def __validation_step_near_field(self, batch, batch_idx, is_test :bool=False):
        x, sources_num, angles, ranges = self.__prepare_batch_near_field(batch)
        angles_pred, ranges_pred, source_estimation, _ = self(x, sources_num)
        loss = self.validation_loss(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        if is_test:
            _, loss_angle, loss_range = self.test_loss_separated(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)
            loss = (loss, loss_angle, loss_range)
        return loss, acc

    def __test_step_near_field(self, batch, batch_idx):
        return self.__validation_step_near_field(batch, batch_idx, is_test=True)

    def __prepare_batch_near_field(self, batch):
        x, sources_num, labels = batch
        validate_constant_sources_number(sources_num)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        x = x.to(self.device)
        angles = angles.to(self.device)
        ranges = ranges.to(self.device)
        return x, sources_num, angles, ranges

    def __prediction_step_near_field(self, batch, batch_idx):
        x = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        angles_pred, distance_pred, _, _ = self(x)
        return angles_pred, distance_pred

    def __set_diff_method(self, diff_method: str, system_model: SystemModel):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if self.field_type == "far":
            if diff_method.startswith("root_music"):
                self.diff_method = RootMusic(system_model=system_model, model_order_estimation=self.regularization)
            elif diff_method.startswith("esprit"):
                self.diff_method = ESPRIT(system_model=system_model, model_order_estimation=self.regularization)
            elif diff_method.endswith("music_1d"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle", model_order_estimation=self.regularization)
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} scenario")
        elif self.field_type == "near":
            if diff_method.endswith("music_2D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle, range", model_order_estimation=self.regularization)
            elif diff_method.endswith("music_1d"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="range")
            elif diff_method.endswith("beamformer"):
                self.diff_method = Beamformer(system_model=system_model)
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} Field scenario")

    def __set_criterion(self, train_loss_type: str = None):
        if self.field_type == "far":
            if train_loss_type == "rmspe":
                self.train_loss = RMSPELoss()
            elif train_loss_type == "music_spectrum":
                self.train_loss = MusicSpectrumLoss(system_model=self.system_model)
            else:
                raise ValueError(f"SubspaceNet.set_criterion: Unrecognized loss type: {train_loss_type}")
            self.validation_loss = RMSPELoss()
            self.test_loss = RMSPELoss()

        elif self.field_type == "near":
            if train_loss_type == "rmspe":
                self.train_loss = CartesianLoss()
            elif train_loss_type == "music_spectrum":
                self.train_loss = MusicSpectrumLoss(system_model=self.system_model)
            else:
                raise ValueError(f"SubspaceNet.set_criterion: Unrecognized loss type: {train_loss_type}")
            self.validation_loss = CartesianLoss()
            self.test_loss = CartesianLoss()
            self.test_loss_separated = RMSPELoss(1.0)

    def __adjust_psd_eps(self):
        if self.training: #
            return self.psd_epsilon
        else:
            return self.psd_epsilon

        snr = self.system_model.params.snr
        is_nf = self._get_name().startswith("NF")
        is_v2 = self.variant == "V2"

        if self.system_model.params.signal_nature == "non-coherent":
            snr_mapping = {
                10: self.psd_epsilon * (
                    1e6 if is_v2 and is_nf else 5e7 if is_nf else 1e11 if is_v2 else 5e8
                ),
                5: self.psd_epsilon * (1e5 if is_v2 and is_nf else 1 / 1.5),
                0: self.psd_epsilon / (1.5 if is_nf else 2),
                -5: self.psd_epsilon / (20 if is_nf else 1.5),
                -10: self.psd_epsilon / 1.5,
            }
        else:  # Coherent case
            snr_mapping = {
                10: self.psd_epsilon * (5e10 if is_v2 and not is_nf else 1e8),
                5: self.psd_epsilon * 1/ 10,
                0: self.psd_epsilon * (5e5 if is_v2 else 1 / 5 if is_nf else 1 / 10),
            }

        return snr_mapping.get(snr, self.psd_epsilon / (2 if is_v2 else 10))
