"""
DeepCNN: Deep learning algorithm as described in:
        [4] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.
"""

import torch
import torch.nn as nn

class DeepCNN(nn.Module):
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

    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X