"""
DA-MUSIC: Deep Augmented MUSIC model-based deep learning algorithm as described in
        [3] J. P. Merkofer, G. Revach, N. Shlezinger, and R. J. van Sloun, “Deep
        augmented MUSIC algorithm for data-driven DoA estimation,” in IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), 2022, pp. 3598-3602."
"""

import torch
import torch.nn as nn

class DeepAugmentedMUSIC(nn.Module):
    """DeepAugmentedMUSIC is a model-based deep learning model for Direction of Arrival (DOA) estimation.

    Attributes:
        N (int): Number of sensors.
        T (int): Number of observations.
        M (int): Number of sources.
        angels (torch.Tensor): Tensor containing angles from -pi/2 to pi/2.
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden layer.
        rnn (nn.GRU): Recurrent neural network module.
        fc (nn.Linear): Fully connected layer.
        fc1 (nn.Linear): Fully connected layer.
        fc2 (nn.Linear): Fully connected layer.
        fc3 (nn.Linear): Fully connected layer.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.
        DropOut (nn.Dropout): Dropout layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer.
        sv (torch.Tensor): Steering vector.

    Methods:
        steering_vec(): Computes the steering vector based on the specified parameters.
        spectrum_calculation(Un: torch.Tensor): Calculates the MUSIC spectrum.
        pre_MUSIC(Rz: torch.Tensor): Applies the MUSIC operation for generating the spectrum.
        forward(X: torch.Tensor): Performs the forward pass of the DeepAugmentedMUSIC model.
    """

    def __init__(self, N: int, T: int, M: int):
        """Initializes the DeepAugmentedMUSIC model.

        Args:
        -----
            N (int): Number of sensors.
            M (int): Number of sources.
            T (int): Number of observations.
        """
        super(DeepAugmentedMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * torch.pi / 2, torch.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
        # Weights initialization
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def steering_vec(self):
        """Computes the ideal steering vector based on the specified parameters.
            equivalent to src.system_model.steering_vec method, but support pyTorch.

        Returns:
        --------
            tensor.Torch: the steering vector
        """
        sv = []
        for angle in self.angels:
            a = torch.exp(
                -1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle)
            )
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un: torch.Tensor):
        spectrum_equation = []
        for i in range(self.angels.shape[0]):
            spectrum_equation.append(
                torch.real(
                    torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i]
                )
            )
        spectrum_equation = torch.stack(spectrum_equation, dim=0)
        spectrum = 1 / spectrum_equation

        return spectrum, spectrum_equation

    def pre_MUSIC(self, Rz: torch.Tensor):
        """Applies the MUSIC operration for generating spectrum

        Args:
            Rz (torch.Tensor): Generated covariance matrix

        Returns:
            torch.Tensor: The generated MUSIC spectrum
        """
        spectrum = []
        bs_Rz = Rz
        for iter in range(self.batch_size):
            R = bs_Rz[iter]
            # Extract eigenvalues and eigenvectors using EVD
            _, eigenvectors = torch.linalg.eig(R)
            # Noise subspace as the eigenvectors which associated with the M first eigenvalues
            Un = eigenvectors[:, self.M:]
            # Calculate MUSIC spectrum
            spectrum.append(self.spectrum_calculation(Un)[0])
        return torch.stack(spectrum, dim=0)

    def forward(self, X: torch.Tensor):
        """
        Performs the forward pass of the DeepAugmentedMUSIC model.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: The estimated DOA.
        """
        # X shape == [Batch size, N, T]
        self.BATCH_SIZE = X.shape[0]
        ## Architecture flow ##
        # decompose X and concatenate real and imaginary part
        X = torch.cat(
            (torch.real(X), torch.imag(X)), 1
        )  # Shape ==  [Batch size, 2N, T]
        # Reshape Output shape: [Batch size, T, 2N]
        X = X.view(X.size(0), X.size(2), X.size(1))
        # Apply batch normalization
        X = self.BatchNorm(X)
        # GRU Clock
        gru_out, hn = self.rnn(X)
        Rx = gru_out[:, -1]
        # Reshape Output shape: [Batch size, 1, 2N]
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1))
        # FC Block
        Rx = self.fc(Rx)  # Shape: [Batch size, 1, 2N^2])
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, : self.N, :]  # Shape [Batch size, N, N])
        Rx_imag = Rx_view[:, self.N:, :]  # Shape [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape [Batch size, N, N])
        # Build MUSIC spectrum
        spectrum = self.pre_MUSIC(Kx_tag)  # Shape [Batch size, 361(grid_size)])
        # Apply peak detection using FC block #2
        y = self.ReLU(self.fc1(spectrum))  # Shape [Batch size, 361(grid_size)])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N)
        # Find doa
        DOA = self.fc3(y)  # Shape [Batch size, M)
        return DOA