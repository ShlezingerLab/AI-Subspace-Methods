"""
DeepRootMUSIC: model-based deep learning algorithm as described in:
        [1] D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun and N. Shlezinger,
        "Deep Root Music Algorithm for Data-Driven Doa Estimation," ICASSP 2023 -
        2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096504.
"""


import torch
import torch.nn as nn

from src.utils import *
from src.methods_pack.root_music import root_music

class DeepRootMUSIC(nn.Module):
    """DeepRootMUSIC is model-based deep learning model for DOA estimation problem.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        LeakyReLU (nn.LeakyReLU): Leaky reLu activation function, with activation_value.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, activation_value: float):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            activation_value (float): Value for the activation function.

        """
        super(DeepRootMUSIC, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(activation_value)
        self.DropOut = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the DeepRootMUSIC.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.LeakyReLU(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        # DCNN block #1
        x = self.deconv1(x)
        x = self.LeakyReLU(x)
        # DCNN block #2
        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv3(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx_tag, eps=1)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots = root_music(
            Rz, self.M, self.batch_size
        )
        return doa_prediction, doa_all_predictions, roots, Rz