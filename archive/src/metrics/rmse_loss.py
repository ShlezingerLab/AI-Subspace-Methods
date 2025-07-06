import torch
import torch.nn as nn
from src.utils import device

class RMSELoss(nn.MSELoss):
    def __init__(self, *args):
        super(RMSELoss, self).__init__(*args)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = super(RMSELoss, self).forward(input.to(device), target.to(device))
        return torch.sqrt(mse_loss)