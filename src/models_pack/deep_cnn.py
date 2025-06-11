"""
DeepCNN: Deep learning algorithm as described in:
        [4] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.
"""

import torch
import torch.nn as nn
from models_pack.parent_model import ParentModel
from src.utils import sample_covariance, validate_constant_sources_number
import numpy as np
from src.metrics import RMSPELoss, CartesianLoss

class DeepCNN(ParentModel):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, system_model):
        ## input dim (N, T)
        super(DeepCNN, self).__init__(system_model)
        self.__init_grid_params()
        self.__set_criterion()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.system_model.params.N - 5) * (self.system_model.params.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.dropOut = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        angle_grid_size = len(self.angles_dict) if hasattr(self, 'angles_dict') else 1
        self.fc_angle = nn.Sequential(
                                    nn.Linear(1024, angle_grid_size),
                                    self.sigmoid
                                )
        if self.system_model.params.field_type.startswith("near"):
            range_grid_size = len(self.ranges_dict) if hasattr(self, 'ranges_dict') else 1
            self.fc_range = nn.Sequential(
                                    nn.Linear(1024, range_grid_size),
                                    self.sigmoid
                                    )

    def forward(self, x):   
        # x is complex with shape [Batch size, N, T]
        X = self.pre_processing(x)  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.relu(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.relu(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.relu(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.relu(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.dropOut(self.relu(self.fc1(X)))  # [Batch size, 4096]
        X = self.dropOut(self.relu(self.fc2(X)))  # [Batch size, 2048]
        X = self.dropOut(self.relu(self.fc3(X)))  # [Batch size, 1024]
        return self.__get_propabilities(X)  # [Batch size, angle_grid_size] or [Batch size, angle_grid_size, range_grid_size]

    def __get_propabilities(self, x):
        if self.system_model.params.field_type.startswith("far"):
            # Far field case, only angle estimation
            return self.fc_angle(x)
        elif self.system_model.params.field_type in ["near", "full"]:
            # Near field case, angle and range estimation
            angle_probs = self.fc_angle(x)
            range_probs = self.fc_range(x)
            return angle_probs, range_probs
        else:
            raise ValueError(f"{self}.__get_propabilities: Unrecognized field type for MUSIC class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")
    
    def training_step(self, batch, batch_idx):
        if self.system_model.params.field_type.startswith("far"):
            x, sources_num, vec_angles, _ = self.__prepare_batch_far_field(batch)
            probs = self(x)
            loss = self.train_loss(probs, vec_angles)
        elif self.system_model.params.field_type in ["near", "full"]:
            x, sources_num, vec_angles, _, vec_ranges, _ = self.prepare_batch_near_field(batch)
            angle_probs, range_probs = self(x)
            loss_angle = self.train_loss(angle_probs, vec_angles)
            loss_range = self.train_loss(range_probs, vec_ranges)
            loss = loss_angle + loss_range
        else:
            raise ValueError(f"{self}.__training_step: Unrecognized field type for DeepCNN class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.system_model.params.field_type.startswith("far"):
            x, sources_num, _, angles = self.__prepare_batch_far_field(batch)
            probs = self(x)
            angles_pred = self.get_labels(probs, sources_num)
            loss = self.test_loss(angles_pred=angles_pred, angles=angles)
        elif self.system_model.params.field_type in ["near", "full"]:
            x, sources_num, _, angles, _, ranges = self.prepare_batch_near_field(batch)
            angle_probs, range_probs = self(x)
            angles_pred = self.get_labels(angle_probs, sources_num)
            ranges_pred = self.get_labels(range_probs, sources_num, is_range=True)
            loss = self.test_loss(angles_pred=angles_pred, angles=angles,
                                 ranges_pred=ranges_pred, ranges=ranges)
            _, loss_angle, loss_range = self.test_loss_separated(angles_pred=angles_pred, angles=angles,
                                                                ranges_pred=ranges_pred, ranges=ranges)
            loss = (loss, loss_angle, loss_range)
        else:
            raise ValueError(f"{self}.__training_step: Unrecognized field type for DeepCNN class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")
        return loss
    
    def get_labels(self, probs, sources_num, is_range=False):
        if not is_range:
            # probs is of shape [Batch size, angle_grid_size]
            angles_pred = torch.zeros(probs.size(0), sources_num, dtype=torch.float64, device=self.device)
            for i in range(probs.size(0)):
                # Get the indices of the top sources_num probabilities
                top_indices = torch.topk(probs[i], sources_num).indices
                # Map the indices to the corresponding angles using angle_to_index
                angles_pred[i] = torch.tensor([self.angles_dict[idx.item()] for idx in top_indices],
                                              dtype=torch.float64, device=self.device)
            return angles_pred
        else:
            # probs is of shape [Batch size, range_grid_size]
            ranges_pred = torch.zeros(probs.size(0), sources_num, dtype=torch.float64, device=self.device)
            for i in range(probs.size(0)):
                # Get the indices of the top sources_num probabilities
                top_indices = torch.topk(probs[i], sources_num).indices
                # Map the indices to the corresponding ranges using range_to_index
                ranges_pred[i] = torch.tensor([self.ranges_dict[idx.item()] for idx in top_indices],
                                              dtype=torch.float64, device=self.device)
            return ranges_pred
    
    def __prepare_batch_far_field(self, batch):
        x, sources_num, angles = super().prepare_batch_far_field(batch)
        # from the angles, create a vector that assigns 1 to the corresponding angle and 0 to others.
        batch_size = x.size(0)
        num_angles = len(self.angles_dict)
        output = torch.zeros(batch_size, num_angles)
        for i in range(batch_size):
            for angle in angles[i]:
                if angle.item() in self.angle_to_index:
                    output[i, self.angle_to_index[angle.item()]] = 1.0
                else:
                    raise ValueError(f"Angle {angle.item()} not found in angle_to_index mapping.")
        return x, sources_num, output, angles

    def prepare_batch_near_field(self, batch):
        x, sources_num, angles, ranges = super().prepare_batch_near_field(batch)
        batch_size = x.size(0)
        num_angles = len(self.angles_dict)
        output_angles = torch.zeros(batch_size, num_angles)
        for i in range(batch_size):
            for angle in angles[i]:
                if angle.item() in self.angle_to_index:
                    output_angles[i, self.angle_to_index[angle.item()]] = 1.0
                else:
                    raise ValueError(f"Angle {angle.item()} not found in angle_to_index mapping.")
        num_ranges = len(self.ranges_dict)
        output_ranges = torch.zeros(batch_size, num_ranges)
        for i in range(batch_size):
            for range_ in ranges[i]:
                if range_.item() in self.range_to_index:
                    output_ranges[i, self.range_to_index[range_.item()]] = 1.0
                else:
                    raise ValueError(f"Range {range_.item()} not found in range_to_index mapping.")
        return x, sources_num, output_angles, angles, output_ranges, ranges

    def pre_processing(self, x):
        """
        Pre-process the input tensor to match the expected input shape for the model,
        which is the empirical covariance real part, imaginary part, and the angle difference between them.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, T).
        
        Returns:
            torch.Tensor: Reshaped input tensor of shape (B, N, N, 3).
        """
        Rx = sample_covariance(x)
        Rx_real = torch.real(Rx)
        Rx_imag = torch.imag(Rx)
        angle_diff = torch.atan2(Rx_imag, Rx_real)  # Calculate angle difference
        # Stack the real part, imaginary part, and angle difference along the last dimension
        Rx_stacked = torch.stack((Rx_real, Rx_imag, angle_diff), dim=-1)
        return Rx_stacked.view(Rx_stacked.size(0), 3, self.N, self.N)  # Reshape to (B, N, N, 3)

    def __init_grid_params(self):
        angle_range = np.deg2rad(self.system_model.params.doa_range)
        angle_resolution = np.deg2rad(self.system_model.params.doa_resolution / 2)
        angle_decimals = int(np.ceil(np.log10(1 / angle_resolution)))

        if self.system_model.params.field_type.startswith("far"):
            # if it's the Far field case, need to init angles range.
            self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution,
                                            dtype=torch.float64).to(torch.float64)
            self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)
            self.angle_to_index = {angle.item(): idx for idx, angle in enumerate(self.angles_dict)}
        elif self.system_model.params.field_type in ["near", "full"]:
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            if self.estimation_params.startswith("angle"):
                self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, angle_resolution,
                                                dtype=torch.float64).to(torch.float64)
                # self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)
                self.angle_to_index = {angle.item(): idx for idx, angle in enumerate(self.angles_dict)}


            if self.estimation_params.endswith("range"):
                fraunhofer_ratio = self.system_model.params.max_range_ratio_to_limit
                distance_resolution = self.system_model.params.range_resolution / 2
                max_distance = min(self.system_model.fraunhofer, fraunhofer * fraunhofer_ratio + distance_resolution)
                self.ranges_dict = torch.arange(np.ceil(fresnel),
                                                max_distance,
                                                distance_resolution, dtype=torch.float64)
                self.range_to_index = {range_.item(): idx for idx, range_ in enumerate(self.ranges_dict)}
        else:
            raise ValueError(f"{self}.__define_grid_params: Unrecognized field type for MUSIC class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")
    
    def __set_criterion(self):
        self.train_loss = nn.BCELoss(reduction='none')
        if self.field_type == "far":
            self.test_loss = RMSPELoss()
        elif self.field_type == "near":
            self.test_loss = CartesianLoss()
            self.test_loss_separated = RMSPELoss(1.0)
    
    def __str__(self):
        return f"DeepCNN(N={self.system_model.params.N}, angle_grid_size={self.fc_angle.out_features}, range_grid_size={self.fc_range.out_features if hasattr(self, 'fc_range') else 'N/A'})"
    
    def __repr__(self):
        return self.__str__()