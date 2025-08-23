from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel
import torch

class NFSubspaceNet(SubspaceNet):
    def __init__(self, tau: int, system_model: SystemModel=None, regularization: str=None, variant: str="small",
                  norm_layer: bool=True, psd_epsilon: float=1e-6, batch_norm: bool=False, skip_connection: bool=False,
                  skip_connection_alpha: float=None, initialize_eigenregularization_weight: float=1e-1):
        diff_method = "2d_music"
        train_loss_type = "music_spectrum"
        field_type = "near"
        super(NFSubspaceNet, self).__init__(tau, diff_method, train_loss_type, system_model, field_type,
                                            regularization, variant, norm_layer, psd_epsilon, batch_norm,
                                            skip_connection, skip_connection_alpha, initialize_eigenregularization_weight)

    def forward(self, x: torch.Tensor, sources_num: torch.tensor = None):
        return super(NFSubspaceNet, self).forward(x, sources_num)