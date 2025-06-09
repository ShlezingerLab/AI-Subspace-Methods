"""
Inspired by the paper:
J. Ji, W. Mao, F. Xi and S. Chen,
"TransMUSIC: A Transformer-Aided Subspace Method for DOA Estimation with Low-Resolution ADCS,"
 ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea,
 Republic of, 2024, pp. 8576-8580, doi: 10.1109/ICASSP48485.2024.10447483. keywords:
  {Direction-of-arrival estimation;Quantization (signal);Acoustic distortion;Simulation;Estimation;Signal processing algorithms
  ;Transformers;DOA estimation;Transformer;low-resolution ADCs;quantization},


"""

import torch
import torch.nn as nn
import math

# Internal imports
from src.system_model import SystemModel
from src.methods_pack.music import MUSIC
from src.models_pack.parent_model import ParentModel
from src.metrics import RMSPELoss, CartesianLoss
from src.config import device

class ShiftedReLU(nn.ReLU):
    def __init__(self, shift=0.5):
        super(ShiftedReLU, self).__init__()
        self.shift = shift

    def forward(self, input):
        return super(ShiftedReLU, self).forward(input) + self.shift


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.device = device

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(self.device) + self.pe[:x.size(0)].to(self.device)

        return x


class TransMUSIC(ParentModel):
    """

    """

    def __init__(self, system_model: SystemModel = None, mode: str = "batch_norm"):
        super(TransMUSIC, self).__init__(system_model)
        self.params = self.system_model.params
        self.N = self.params.N
        self.estimation_params = None
        self.train_mode = "subspace_train"
        if self.params.field_type == "far":
            self.estimation_params = "angle"
            self.rmspe_loss = RMSPELoss()
        elif self.params.field_type == "near":
            self.estimation_params = "angle, range"
            self.rmspe_loss = CartesianLoss()
            self.separated_test_loss = RMSPELoss(1.0)
        else:
            raise ValueError(f"TransMUSIC.__init__: Unrecognized field type {self.params.field_type}")
        self.music = MUSIC(system_model=self.system_model, estimation_parameter=self.estimation_params)
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

        if mode == "batch_norm":
            self.norm = nn.BatchNorm1d(self.N * 2).to(self.device)
        elif mode == "layer_norm":
            self.norm = nn.LayerNorm(normalized_shape=[self.N * 2]).to(self.device)
        else:
            raise ValueError(f"TransMUSIC.__init__: unrecognized mode {mode}")

        self.pos_encoder = PositionalEncoding(self.N * 2)  # Position embedding

        # Define encoder layer
        e_layer = nn.TransformerEncoderLayer(d_model=self.N * 2,
                                             nhead=self.N,  # The number of heads in a multi head attention model
                                             dim_feedforward=1024,  # Dimensions of feedforward network models
                                             dropout=0,  # Dropout value
                                             activation="relu",
                                             layer_norm_eps=1e-05,
                                             batch_first=True,
                                             norm_first=False,
                                             device=None,
                                             dtype=None).to(self.device)

        self.encoder = nn.TransformerEncoder(e_layer,
                                             num_layers=3,
                                             norm=None).to(self.device)

        self.input_linear = nn.Linear(in_features=self.N * 2, out_features=2 * self.N ** 2).to(self.device)
        if self.estimation_params == "angle":
            self.input_dim = self.music.angles_dict.shape[0]
            if isinstance(self.music.system_model.params.M, tuple):
                output_dim = self.music.system_model.params.M[1]
            else:
                output_dim = self.music.system_model.params.M
        elif self.estimation_params == "angle, range":
            self.input_dim = self.music.angles_dict.shape[0] * self.music.ranges_dict.shape[0]
            if isinstance(self.music.system_model.params.M, tuple):
                output_dim = (self.music.system_model.params.M[1] - 1) * 2
            else:
                output_dim = self.music.system_model.params.M * 2
            # self.activation = ShiftedReLU(shift=np.floor(self.music.system_model.fresnel)).to(device)
            # self.activation = nn.ReLU().to(device)
            self.activation = nn.LeakyReLU(negative_slope=0.0001).to(self.device)
        else:
            raise ValueError(f"TransMUSIC.__init__: unrecognized estimation parameter {self.estimation_params}")
        self.output = nn.Sequential(

            nn.Linear(in_features=self.input_dim, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=output_dim)
        ).to(self.device)

        self.source_number_estimator = nn.Sequential(
            nn.Linear(in_features=2 * self.N ** 2, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=32, out_features=self.params.N - 1),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, x):
        N = self.N
        x = self.pre_processing(x)
        # The input X is [size, 16200] batch_ Size=16, input dimension N*2, sequence length T

        size = x.shape[0]  # Get batch size

        x3 = self._get_noise_subspace(x)
        if self.train_mode == "subspace_train":
            x4 = x3.reshape(size, N * 2, N).to(self.device)  # Change its mapping covariance to [size, N * 2, N]
            Un = torch.complex(x4[:, :N, :], x4[:, N:, :]).to(torch.complex128)  # feature vector  [size, N, N]
            spectrum = self.music.get_music_spectrum_from_noise_subspace(Un)  # Calculate spectrum
            x7 = spectrum.float().to(self.device)
            x7 = x7.view(size, -1).to(self.device)  # Change the shape of the spectrum to [size, N * 2]
            predictions = self.output(x7).to(self.device)
            if self.estimation_params == "angle, range":
                angles, distances = torch.split(predictions, predictions.shape[-1] // 2, dim=1)
                distances = self.activation(distances).to(self.device)
                predictions = torch.cat([angles, distances], dim=1).to(self.device)
            with torch.no_grad():
                x9 = x3.detach()
                prob_sources_est = self.source_number_estimator(x9)
        elif self.train_mode == "num_source_train":
            prob_sources_est = self.source_number_estimator(x3)
            with torch.no_grad():
                x4 = x3.detach().reshape(size, N * 2, N).to(self.device)  # Change its mapping covariance to [size, N * 2, N]
                Un = torch.complex(x4[:, :N, :], x4[:, N:, :]).to(torch.complex32)  # feature vector  [size, N, N]
                spectrum = self.music.get_music_spectrum_from_noise_subspace(Un)  # Calculate spectrum
                x7 = spectrum.float().to(self.device)
                x7 = x7.view(size, -1).to(self.device)  # Change the shape of the spectrum to [size, N * 2]
                predictions = self.output(x7).to(self.device)
                if self.estimation_params == "angle, range":
                    angles, distances = torch.split(predictions, predictions.shape[-1] // 2, dim=1)
                    distances = self.activation(distances).to(self.device)
                    predictions = torch.cat([angles, distances], dim=1).to(self.device)
        else:
            raise ValueError(f"TransMUSIC.forward: Unrecognized {self.train_mode}")
        return predictions, prob_sources_est

    def _get_noise_subspace(self, x):
        x = self.norm(x.to(torch.float32)).to(self.device)  # Become [size, N * 2, T]

        x = x.permute(2, 0, 1).float().to(self.device)  # Exchange dimension becomes [T, size, N * 2]

        # Position embedding
        x = self.pos_encoder(x.to(self.device)).to(self.device)  # x: Tensor, shape [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2).float().to(self.device)  # Exchange dimension becomes [size, T, N * 2]

        x1 = self.encoder(x.to(self.device))  # Transformer_ Encoder network output becomes [size, T, N * 2]
        x2 = torch.mean(x1, dim=1)  # Output becomes [size, N * 2]
        x3 = self.input_linear(x2).to(self.device)
        return x3

    def pre_processing(self, x: torch.Tensor):
        """
        The input data is a complex signal of size [batch_size, N, T] and the input of the model need to be real signal
        of size [batch_size, 2N, T]
        """
        if x.dim() == 2:
            x = x[None, :, :]
        x = torch.cat([x.real, x.imag], dim=1)
        return x

    def update_train_mode(self, train_mode: str):
        if train_mode not in ["subspace_train", "num_source_train"]:
            raise ValueError(f"TransMUSIC.update_train_mode: Unrecognized train_mode {train_mode}")
        self.train_mode = train_mode

    def print_model_params(self):
        return f"in_dim={self.input_dim}"

    def get_model_params(self):
        return {"in_dim": self.input_dim}

    def training_step(self, batch, batch_idx):
        if self.params.field_type == "far":
            return self.__training_step_far_field(batch, batch_idx)
        elif self.params.field_type == "near":
            return self.__training_step_near_field(batch, batch_idx)

    def validation_step(self, batch, batch_idx, is_test: bool=False):
        if self.params.field_type == "far":
            return self.__valid_step_far_field(batch, batch_idx)
        elif self.params.field_type == "near":
            return self.__valid_step_near_field(batch, batch_idx, is_test)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, is_test=True)

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def __training_step_far_field(self, batch, batch_idx):
        x, sources_num, angles = batch
        x = x.to(self.device)
        angles = angles.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"TransMUSIC.__training_step_far_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles_pred, prob_source_number = self(x)
        angles_pred = angles_pred[:, :angles.shape[1]]
        if self.train_mode == "num_source_train":
            # calculate the cross entropy loss for the source number estimation
            one_hot_sources_num = (nn.functional.one_hot(sources_num, num_classes=prob_source_number.shape[1])
                                   .to(self.device).to(torch.float32))
            loss = self.ce_loss(prob_source_number, one_hot_sources_num.repeat(
                prob_source_number.shape[0], 1)) * x.shape[0]
        else:
            loss = self.rmspe_loss(angles_pred=angles_pred, angles=angles)
        # calculate the source estimation
        source_estimation = torch.argmax(prob_source_number, dim=1)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc, None

    def __training_step_near_field(self, batch, batch_idx):
        x, sources_num, labels = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"SubspaceNet.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        x = x.requires_grad_(True).to(self.device)
        angles = angles.requires_grad_(True).to(self.device)
        ranges = ranges.requires_grad_(True).to(self.device)
        model_output, prob_source_number = self(x)
        if self.train_mode == "num_source_train":
            # calculate the cross entropy loss for the source number estimation
            one_hot_sources_num = (nn.functional.one_hot(sources_num, num_classes=prob_source_number.shape[1])
                                   .to(self.device).to(torch.float32))
            loss = self.ce_loss(prob_source_number, one_hot_sources_num.repeat(
                prob_source_number.shape[0], 1)) * x.shape[0]
        else:
            angles_pred, ranges_pred = torch.split(model_output, model_output.shape[1] // 2, dim=1)
            angles_pred = angles_pred[:, :angles.shape[1]]
            ranges_pred = ranges_pred[:, :ranges.shape[1]]
            loss = self.rmspe_loss(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)
        loss = torch.sum(loss)  # Sum the loss over the batch
        source_estimation = torch.argmax(prob_source_number, dim=1)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc, None

    def __valid_step_near_field(self, batch, batch_idx, is_test: bool=False):
        x, sources_num, labels = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"TransMUSIC.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        x = x.to(self.device)
        angles = angles.to(self.device)
        ranges = ranges.to(self.device)
        model_output, prob_source_number = self(x)
        if self.train_mode == "num_source_train":
            # calculate the cross entropy loss for the source number estimation
            one_hot_sources_num = (nn.functional.one_hot(sources_num, num_classes=prob_source_number.shape[1])
                                   .to(self.device).to(torch.float32))
            loss = self.ce_loss(prob_source_number, one_hot_sources_num.repeat(
                prob_source_number.shape[0], 1)) * x.shape[0]
        else:
            angles_pred, ranges_pred = torch.split(model_output, model_output.shape[1] // 2, dim=1)
            angles_pred = angles_pred[:, :angles.shape[1]]
            ranges_pred = ranges_pred[:, :ranges.shape[1]]
            loss = self.rmspe_loss(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)
        if is_test:
            _, angle_loss, range_loss = self.separated_test_loss(angles_pred=angles_pred, angles=angles, ranges_pred=ranges_pred, ranges=ranges)
            loss = (loss, angle_loss, range_loss)
        source_estimation = torch.argmax(prob_source_number, dim=1)
        acc = self.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc
