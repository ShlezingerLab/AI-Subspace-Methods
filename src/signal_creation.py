import numpy as np
import torch
from random import sample

class SystemModelParams:
    """
    Parameters for the for the simulation at one claass.
    """

    def __init__(self, **kwargs):
        kwargs = {**kwargs}
        for key, value in kwargs.items():
            if isinstance(value, str):
                value = value.lower()
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"SystemModelParams({', '.join(attrs)})"
    

class SystemModel:
    """
    Simplified SystemModel class for 
    far-field
    non-coherent
    narrow-band
    DoA estimation
    """

    def __init__(self, system_model_params: SystemModelParams):
        self.params = system_model_params
        self.array = None
        self.dist_array_elems = None
        self.gain_petrurbation = np.ones(self.params.N, dtype=np.complex64)  # Default gain perturbation
        
        # Set inter-element spacing
        self.define_scenario_params()
        # Initialize array geometry
        self.create_array()



    def define_scenario_params(self):
        """Simplified parameter definition for narrowband far-field"""
        # Distance between array elements (half wavelength spacing)
        self.dist_array_elems = self.params.wavelength / 2

    def create_array(self):
        """
        Create the antenna array geometry
        Location perturbation are applied here"""
        N = self.params.N
        # Linear array: [0, 1, 2, ..., N-1] * (λ/2)
        base_array = np.arange(N, dtype=float) * self.dist_array_elems
        # add a perturbation sampled uniformly between
        # -self.params.location_perturbation and self.params.location_perturbation for middle elements,
        # for the first element sampled uniformly between 0 and self.params.location_perturbation,
        # and for the last element sampled uniformly between -self.params.location_perturbation and 0.
        # This makes sure that the over all size is no more than (N-1)*\lambda/2
        if self.params.location_perturbation is not None:
            first_element_perturbation = np.random.uniform(0, self.params.location_perturbation)
            mid_elements_perturbation = np.random.uniform(-self.params.location_perturbation, 
                                             self.params.location_perturbation, N-2)
            last_element_perturbation = np.random.uniform(-self.params.location_perturbation, 0)
            perturbation = np.concatenate(([first_element_perturbation], mid_elements_perturbation,
                                            [last_element_perturbation]), axis=0)
            base_array += perturbation
        
        self.array = torch.from_numpy(base_array).to(torch.float64)  # Convert to torch tensor
    
    def steering_vec(self, angles: np.ndarray) -> torch.Tensor:
        """
        Compute steering vector for far-field sources
        
        Args:
            angles: Array of source angles in radians
            
        Returns:
            Complex steering matrix of shape (N, M)
        """
        return self.steering_vec_far_field(angles)
    
    #TODO: Check that the steering matrix is implemented correctly for the far-field case
    
    def steering_vec_far_field(self, angles: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Compute far-field steering vectors
        
        Args:
            angles: Array of source angles in RADIANS!!!! (M,)
            
        Returns:
            Complex steering matrix (N, M) including gain and location perturbations if specified
        """
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        
        # Convert to float64 for precision
        angles = angles.to(torch.float64)
        array = self.array.view(-1,1)  # (N, 1)
        
        # Ensure angles is 2D: (1, M) for broadcasting
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)  # (1, M)
        
        
        # Phase delays: (N, 1) * sin(angles) -> (N, M)
        time_delay = array @ torch.sin(angles)  # Broadcasting: (N,1) * (1,M) -> (N,M)
        
        # Steering matrix: complex exponentials
        steering_matrix = torch.exp(-2j * torch.pi * time_delay / self.params.wavelength).to(torch.complex64)  # (N, M)

        if self.params.gain_perturbation_var > 0.0:
            gain_impairments = (torch.ones(self.params.N, dtype=torch.complex64) +
                            torch.randn(self.params.N, dtype=torch.complex64) * torch.sqrt(torch.tensor(self.params.gain_perturbation_var)))
            
            self.gain_petrurbation = gain_impairments

            #make a diagonal matrix of the gain impairments
            diag_gain_impairments = torch.diag(gain_impairments)
            normalization_factor = 1/torch.linalg.norm(gain_impairments, 2, dtype=torch.complex64)
            # Apply gain perturbations
            steering_matrix = normalization_factor * diag_gain_impairments @ steering_matrix

        return steering_matrix

    def get_array(self):
        """
        Get the current array geometry
        
        Returns:
            Numpy array of antenna positions
        """
        return self.array

    def get_antenna_gains(self):
        """
        Get the gain perturbation applied to the steering vector
        
        Returns:
            Numpy array of gain perturbations
        """
        return self.gain_petrurbation if self.gain_petrurbation is not None else np.ones(self.params.N, dtype=torch.complex64)

class Samples(SystemModel):
    """
    Simplified Samples class for signal and noise generation
    Inherits from SystemModel for steering vector computation
    
    Removed:
    - Distance handling (near-field)
    - Coherent signal support
    - Broadband signal support
    - Variable M support within single sample
    """
    
    def __init__(self, system_model_params: SystemModelParams):
        super().__init__(system_model_params)
        self.angles = None
    
    def set_labels(self, angles: list | None = None):
        """
        Set the angles for the sources
        
        Args:
            angles: List of angles in degrees, or None for random generation
        """
        self.set_angles(angles)
    
    def get_labels(self):
        """Get the current source angles as tensor"""
        return torch.tensor(self.angles, dtype=torch.float32)
    
    def set_angles(self, doa: list | None = None):
        """
        Set direction of arrival angles
        
        Args:
            doa: List of angles in degrees, or None for random generation
        """
        if doa is None:
            # Generate random angles with minimum separation
            self.angles = np.deg2rad(self._create_doa_with_gap(gap=10))
        else:
            # Use provided angles
            self.angles = np.deg2rad(np.array(doa))
    
    def _create_doa_with_gap(self, gap: float = 10):
        """
        Create angles with minimum separation (same logic as original)
        
        Args:
            gap: Minimum separation in degrees
            
        Returns:
            List of angles in degrees
        """
        M = self.params.M
        doa_range = self.params.doa_range # 0-90 degrees
        doa_resolution = self.params.doa_resolution 
        
        # Compute effective range - reserving slots for the M-1 gaps between the M sources for allocation
        max_offset = (gap - 1) * (M - 1)
        effective_range = 2 * doa_range - max_offset
        
        if effective_range <= 0:
            raise ValueError(f"Cannot fit {M} sources with {gap}° separation in ±{doa_range}° range")
        
        # Sample positions
        valid_range = np.arange(0, effective_range, doa_resolution)
        sampled_values = sorted(sample(valid_range.tolist(), M))
        
        # Convert to actual angles - put a gap of at least `gap` degrees between sources
        # and shift the angles to start from -doa_range
        DOA = [(gap - 1) * i + x - doa_range for i, x in enumerate(sampled_values)]
        
        return np.round(DOA, 3)
    
    def samples_creation(self, 
                        noise_mean: float = 0, 
                        noise_variance: float = 1,
                        signal_mean: float = 0, 
                        signal_variance: float = 1):
        """
        Create observation samples: X = A @ S + N
        The noise variance is as desribed above and the signal variance is multiplied by the SNR in linear scale.
        
        Args:
            noise_mean: Mean of noise (typically 0)
            noise_variance: Variance of noise (typically 1) 
            signal_mean: Mean of signal (typically 0)
            signal_variance: Variance of signal (typically 1)
            
        Returns:
            tuple: (samples, signal, steering_matrix, noise)
                - samples: Complex observations (N, T)
                - signal: Complex source signals (M, T) 
                - steering_matrix: Complex steering matrix (N, M)
                - noise: Complex noise (N, T)
        """
        # Generate source signals (non-coherent)
        signal = self.signal_creation(signal_mean, signal_variance)
        signal = torch.from_numpy(signal).to(torch.complex64)
        
        # Generate noise
        noise = self.noise_creation(noise_mean, noise_variance)  
        noise = torch.from_numpy(noise).to(torch.complex64)
        
        # Compute steering matrix for current angles
        A = self.steering_vec(self.angles)
        
        # Create observations: X = A @ S + N
        samples = (A @ signal) + noise
        
        return samples, signal, A, noise
    
    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1):
        """
        Generate non-coherent source signals
        
        Args:
            signal_mean: Mean of signals
            signal_variance: Variance of signals
            
        Returns:
            Complex signal matrix (M, T)
        """
        M, T = self.params.M, self.params.T
        
        # Convert SNR from dB to linear scale
        amplitude = np.sqrt(10 ** (self.params.snr / 10)) #NOTE: the SNR is the power ration between the variances, the sqrt is needed than to convert to std.
        
        # Generate M independent complex Gaussian signals
        signals = (amplitude * (np.sqrt(2) / 2) * np.sqrt(signal_variance) * 
                  (np.random.randn(M, T) + 1j * np.random.randn(M, T)) + signal_mean)
        
        return signals
    
    def noise_creation(self, noise_mean: float = 0, noise_variance: float = 1):
        """
        Generate complex white Gaussian noise
        
        Args:
            noise_mean: Mean of noise
            noise_variance: Variance of noise
            
        Returns:
            Complex noise matrix (N, T)
        """
        N, T = self.params.N, self.params.T
        
        # Generate complex white Gaussian noise
        noise = (np.sqrt(noise_variance) * (np.sqrt(2) / 2) * 
                (np.random.randn(N, T) + 1j * np.random.randn(N, T)) + noise_mean)
        
        return noise


def create_single_sample(system_model_params: SystemModelParams, 
                        true_doa: list = None):
    """
    Simplified version of create_dataset for single sample generation
    
    Args:
        system_model_params: System model parameters
        true_doa: Predefined angles in degrees, or None for random
        
    Returns:
        tuple: (observations, labels, samples_model)
            - observations: Complex matrix (N, T)
            - labels: Source angles in radians
            - samples_model: The Samples object used
    """
    # Create samples model
    samples_model = Samples(system_model_params)
    
    # Set source angles
    samples_model.set_labels(true_doa)
    
    # Generate observations
    X, signal, A, noise = samples_model.samples_creation(
        noise_mean=0, noise_variance=1, 
        signal_mean=0, signal_variance=1
    )
    
    # Get ground truth labels
    Y = samples_model.get_labels()
    
    return X, Y, samples_model


