import torch
import torch.nn as nn
from src.system_model import SystemModel
from src.config import device

class MusicSpectrumLoss(nn.Module):
    def __init__(self, system_model: SystemModel, mode:str = "inverse_spectrum",
                 aggregate: str = "sum"):
        super(MusicSpectrumLoss, self).__init__()
        self.device = device
        self.system_model = system_model
        self.array = torch.from_numpy(system_model.array).to(torch.float32).to(self.device).unsqueeze(-1)
        self.number_sensors = system_model.params.N
        self.sensors_distance = system_model.params.wavelength / 2
        if mode not in ["spectrum", "inverse_spectrum"]:
            raise Exception(f"MusicSpectrumLoss: mode {mode} is not defined")
        self.mode = mode
        self.aggregate = aggregate


    def forward(self, **kwargs):
        if "ranges" in kwargs:
            return self.__forward_with_ranges(**kwargs)
        else:
            return self.__forward_without_ranges(**kwargs)

    def __forward_without_ranges(self, **kwargs):
        angles = kwargs["angles"].unsqueeze(-1)
        time_delay = torch.einsum("nm, ban -> ban",
                                  self.array,
                                  torch.sin(angles).repeat(1, 1, self.number_sensors) * self.sensors_distance)
        search_grid = torch.exp(-2 * 1j * torch.pi * time_delay / self.system_model.params.wavelength)
        var1 = torch.bmm(search_grid.conj(), kwargs["noise_subspace"].to(torch.complex128))
        inverse_spectrum = torch.norm(var1, dim=-1)
        if self.mode == "inverse_spectrum":
            loss = torch.sum(inverse_spectrum, dim=1).sum()
        elif self.mode == "spectrum":
            loss = -torch.sum(1 / inverse_spectrum, dim=1).sum()
        return loss

    def __forward_with_ranges(self, **kwargs):
        angles = kwargs["angles"][:, :, None]
        ranges = kwargs["ranges"][:, :, None].to(torch.float64)
        array_square = torch.pow(self.array, 2).to(torch.float64)
        noise_subspace = kwargs["noise_subspace"].to(torch.complex128)
        first_order = torch.einsum("nm, bna -> bna",
                                   self.array,
                                   torch.sin(angles).repeat(1, 1, self.number_sensors).transpose(1, 2) * self.sensors_distance)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(angles) * self.sensors_distance, 2),
                                        ranges)
        second_order = second_order.repeat(1, 1, self.number_sensors)
        second_order = torch.einsum("nm, bna -> bna", array_square, second_order.transpose(1, 2))

        time_delay = first_order + second_order

        search_grid = torch.exp(2 * -1j * torch.pi * time_delay / self.system_model.params.wavelength)
        var1 = torch.einsum("bak, bkl -> bal",
                            search_grid.conj().transpose(1, 2)[:, :, :noise_subspace.shape[1]],
                            noise_subspace)
        # get the norm value for each element in the batch.
        inverse_spectrum = torch.norm(var1, dim=-1) ** 2
        if self.mode == "inverse_spectrum":
            loss = torch.sum(inverse_spectrum, dim=-1)
        elif self.mode == "spectrum":
            loss = -torch.sum(1 / inverse_spectrum, dim=-1)
        else:
            raise Exception(f"MusicSpectrumLoss: mode {self.mode} is not defined")

        if self.aggregate == "sum":
            return torch.sum(loss) # TODO
        elif self.aggregate == "mean":
            return torch.mean(loss)
        else:
            return loss

    def __str__(self):
        return "music_spectrum"