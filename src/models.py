"""Subspace-Net 
Details
----------
Name: models.py
Authors: Dor Haim Shmuel, Arad Gast
Created: 01/10/21
Edited: 29/05/24
"""

# Imports
import warnings
import torch.nn as nn
# Internal Imports
from src.system_model import SystemModel, SystemModelParams
from src.models_pack import TransMUSIC, SubspaceNet, DCDMUSIC, DeepAugmentedMUSIC, DeepCNN, DeepRootMUSIC, NFSubspaceNet


class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    Attributes:
            model_type(str): The network type
            system_model_params(SystemModelParams): the system model parameters.
            model_params(dict): The parameters for the model.
            model(nn.Module) : An instance of the model
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type: str = str(None)
        self.system_model: SystemModel = None
        self.model_params: dict = None
        self.model: nn.Module = None
        self.samples_size: int = None

    def set_model_type(self, model_type: str) -> "ModelGenerator":
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelGenerator: The updated ModelGenerator object.

        Raises:
            ValueError: If model type is not provided.
        """
        self.model_type = model_type
        return self

    def set_system_model(self, system_model_params: SystemModelParams) -> "ModelGenerator":
        """
        Set the system model.

        Parameters:
            system_model (SystemModel): The system model.

        Returns:
            ModelGenerator: The updated ModelParams object.

        Raises:
            ValueError: If system_model is not provided.
        """
        self.system_model = SystemModel(system_model_params, nominal=True)
        return self

    def set_model_params(self, model_params: dict):
        """
        Set the model params.

        Parameters:
            model_params (dict): The model's parameters.

        Returns:
            ModelGenerator: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        # verify params for model.
        try:
            self.__verify_model_params(model_params)
        except Exception as e:
            raise ValueError(f"ModelGenerator.set_model_params: {e}")
        self.model_params = model_params
        return self

    def set_samples_size(self, samples_size: int) -> "ModelGenerator":
        """
        Set the samples size for checkpoint naming when dataset size varies.

        Parameters:
            samples_size (int): The dataset size.

        Returns:
            ModelGenerator: The updated ModelGenerator object.
        """
        self.samples_size = samples_size
        return self

    def set_model(self) -> "ModelGenerator":
        """
        Set the model based on the model type and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the model type is not defined.
        """
        if self.model_type.startswith("DA-MUSIC"):
            self.__set_da_music()
        elif self.model_type.startswith("DR_MUSIC"):
            self.__set_dr_music()
        elif self.model_type.startswith("DeepCNN"):
            self.__set_deepcnn()
        elif self.model_type.startswith("SubspaceNet"):
            self.__set_subspacenet()
        elif self.model_type.startswith("NF-SubspaceNet"):
            self.__set_nf_subspacenet()
        elif self.model_type.startswith("DCD-MUSIC"):
            self.__set_dcd_music()
        elif self.model_type.startswith("TransMUSIC"):
            self.__set_transmusic()
        else:
            raise Exception(f"ModelGenerator.set_model: Model type {self.model_type} is not defined")

        # Set samples_size on the model if it was specified (for checkpoint naming)
        if self.samples_size is not None and hasattr(self.model, 'samples_size'):
            self.model.samples_size = self.samples_size

        return self

    def __set_subspacenet(self):
        """

        """
        self.model = SubspaceNet(system_model=self.system_model, **self.model_params)
    
    def __set_nf_subspacenet(self):
        """

        """
        self.model = NFSubspaceNet(system_model=self.system_model, **self.model_params)

    def __set_dcd_music(self):
        """

        """
        self.model = DCDMUSIC(system_model=self.system_model, **self.model_params)

    def __set_transmusic(self):
        """

        """
        self.model = TransMUSIC(system_model=self.system_model)

    def __set_deepcnn(self):
        """

        """
        self.model = DeepCNN(system_model=self.system_model)

    def __set_da_music(self):
        """

        """
        N = self.system_model.params.N
        T = self.system_model.params.T
        M = self.system_model.params.M
        self.model = DeepAugmentedMUSIC(N=N, T=T, M=M)

    def __set_dr_music(self):
        """

        """
        self.model = DeepRootMUSIC(**self.model_params)

    def __verify_model_params(self, model_params):
        """
        There are different models, and each one has different set of parameters to set.
        This function just verify the correctness of the params depending on the model.
        """

        if self.model_type.lower() == "subspacenet":
            self.__verify_subspacenet_params(model_params)
        elif self.model_type.lower().startswith("nf-subspacenet"):
            self.__verify_nf_subspacenet_params(model_params)
        elif self.model_type.lower().startswith("dcd-music"):
            self.__verify_dcdmuisc_params(model_params)
        elif self.model_type.lower() == "transmusic":
            self.__verify_transmusic_params(model_params)
        else:
            warnings.warn(f"ModelGenerator.__verify_model_params:"
                             f" currently there is no verification support for {self.model_type}")

    def __verify_transmusic_params(self, model_params):
        """

        """
        pass

    def __verify_subspacenet_params(self, model_params):
        """
        tau: int, diff_method: str = "root_music", field_type: str = "Far"
        """
        self.__verify_tau(model_params.get("tau"))

        field_type = model_params.get("field_type")
        if not isinstance(field_type, str) or not (field_type.lower() in ["far", "near"]):
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f"field_type has to be a str and the possible values are far or near.")

        diff_method = model_params.get("diff_method")
        if isinstance(diff_method, str):
            if field_type.lower() == "far":
                if not (diff_method.lower() in ["root_music", "esprit", "music_1d"]):
                    raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                                     f"for Far field possible diff methods are: root_music, esprit or music_1d")
            else:  # field_type.lower() == "near":
                if not (diff_method.lower() in ["music_1d", "music_2d"]):
                    raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                                     f"for Near field possible diff methods are: music_2d or music_1d")
        else:
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f"field type was not given as a model param.")

        train_loss_type = model_params.get("train_loss_type")
        if not isinstance(train_loss_type, str) or not (train_loss_type.lower() in ["rmspe", "music_spectrum"]):
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f"train_loss_type has to be a str and the possible values are rmspe or music_spectrum.")

        self.__verify_regularization(model_params.get("regularization"))

    def __verify_nf_subspacenet_params(self, model_params):
        """
        tau: int
        diff_method: tuple
        """
        self.__verify_tau(model_params.get("tau"))
        diff_method = model_params.get("diff_method")
        # diff methos has to be 2d_music
        if not isinstance(diff_method, str) or not (diff_method.lower() in ["2d_music"]):
            raise ValueError(f"ModelGenerator.__verify_nf_subspacenet_params:"
                             f"diff_method has to be a str and the possible values are 2d_music.")
        self.__verify_regularization(model_params.get("regularization"))


    def __verify_dcdmuisc_params(self, model_params):
        """
        tau: int
        diff_method: tuple
        """
        self.__verify_tau(model_params.get("tau"))
        diff_method = model_params.get("diff_method")
        if not isinstance(diff_method, tuple) or not (len(diff_method) == 2):
            raise ValueError(f"ModelGenerator.__verify_dcdmuisc_params:"
                             f" diff_method has to be a tuple of two elements.")
        if not (diff_method[0].lower() in ["esprit", "music_1d"]):
            raise ValueError(f"ModelGenerator.__verify_dcdmuisc_params:"
                             f" first element of diff_method has to be music_1D_noise_ss, esprit or music_1d")
        if not (diff_method[1].lower() in ["music_1d"]):
            raise ValueError(f"ModelGenerator.__verify_dcdmuisc_params:"
                             f" second element of diff_method has to be music_1D_noise_ss, esprit or music_1d")
        self.__verify_regularization(model_params.get("regularization"))

    def __str__(self):
        return f"{self.model.get_model_name()}"
    
    def __verify_regularization(self, regularization):
        if regularization is not None and (not isinstance(regularization, str) or regularization.lower() not in ["threshold", "mdl", "aic"]):
            raise ValueError(f"ModelGenerator.__verify_regularization:"
                             f"regularization has to be a str and the possible values are threshold, mdl or aic, or None")
    def __verify_tau(self, tau):
        if not isinstance(tau, int) or not (tau < self.system_model.params.T):
            raise ValueError(f"ModelGenerator.__verify_tau:"
                             f" Tau has to be an int and smaller than T")