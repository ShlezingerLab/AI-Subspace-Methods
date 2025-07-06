import torch
import torch.nn as nn
from src.utils import device

class BeamFromingLoss(nn.Module):
    def __init__(self ,array: torch.Tensor, sensors_distance: float, aggregate: str = "sum"):
        super(BeamFromingLoss, self).__init__()
        self.array = array
        self.sensors_distance = sensors_distance
        self.number_sensors = array.shape[0]
        self.aggregate = aggregate

    def forward(self, **kwargs):
        angles = kwargs["angles"].unsqueeze(-1)
        ranges = kwargs["ranges"].unsqueeze(-1)
        array_square = torch.pow(self.array, 2).to(torch.float64)
        covariance = kwargs["covariance"].to(torch.complex128)

        phase_first_order = torch.einsum("nm, bna -> bna",
                                      self.array,
                                      torch.sin(angles).repeat(1, 1, self.number_sensors).transpose(1, 2) * self.sensors_distance)
        phase_second_order = -0.5 * torch.div(torch.pow(torch.cos(angles) * self.sensors_distance, 2),
                                        ranges)
        phase_second_order = phase_second_order.repeat(1, 1, self.number_sensors)
        phase_second_order = torch.einsum("nm, bna -> bna", array_square, phase_second_order.transpose(1, 2))
        phase = phase_first_order + phase_second_order
        steering_vector = torch.exp(1j * torch.pi * phase)
        beamforming = torch.einsum("bak, bkl -> bal", steering_vector.conj().transpose(1, 2), covariance)
        beamforming = torch.einsum("ban, bna -> ba", beamforming, steering_vector)
        beamforming = torch.real(beamforming.sum(dim=-1)) / self.number_sensors

        if self.aggregate == "sum":
            loss = torch.sum(beamforming)
        elif self.aggregate == "mean":
            loss = torch.mean(beamforming)
        else:
            loss = beamforming

        return -loss