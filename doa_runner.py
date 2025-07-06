"""
DoA Algorithm Runner - Handles training and evaluation of DoA estimation algorithms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import time
from typing import Dict, Any, Optional, Tuple, List

from src.utils import sample_covariance


class AlgorithmFactory:
    """
    Factory class for creating DoA algorithms and training components
    Encapsulated within DoARunner to avoid circular imports and organize related functionality
    """
    
    def __init__(self, device):
        self.device = device
    
    def create_algorithm(self, model_type: str, system_model_params, **kwargs):
        """
        Create DoA estimation algorithm based on model type
        
        Args:
            model_type: Type of algorithm ("diffMUSIC", "MUSIC", etc.)
            system_model_params: System model parameters
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Algorithm instance
        """
        from src.diffmusic import DiffMUSIC
        
        if model_type.lower() == "diffmusic":
            algorithm = DiffMUSIC(
                system_model_params=system_model_params,
                N=system_model_params.N,
                model_order_estimation=kwargs.get('model_order_estimation', None),
                physical_array=kwargs.get('physical_array', None),
                physical_gains=kwargs.get('physical_gains', None)
            )
            algorithm.train()
        elif model_type.lower() == "music":
            # For now, use DiffMUSIC in eval mode for classical MUSIC
            algorithm = DiffMUSIC(
                system_model_params=system_model_params,
                N=system_model_params.N,
                model_order_estimation=kwargs.get('model_order_estimation', None),
                physical_array=kwargs.get('physical_array', None),
                physical_gains=kwargs.get('physical_gains', None)
            )
            algorithm.eval()  # Set to evaluation mode for classical MUSIC behavior
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return algorithm.to(self.device)
    
    def create_loss_function(self, loss_type: str, **kwargs):
        """
        Create loss function based on type
        
        Args:
            loss_type: Type of loss ("rmspe", "spectrum", "unsupervised")
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Loss function instance
        """
        from src.diffmusic import DiffMUSICLoss
        
        return DiffMUSICLoss(loss_type=loss_type, **kwargs)
    
    def create_optimizer(self, algorithm, optimizer_type: str, learning_rate: float, **kwargs):
        """
        Create optimizer for algorithm parameters
        
        Args:
            algorithm: Algorithm instance with learnable parameters
            optimizer_type: Type of optimizer ("Adam", "SGD")
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer instance
        """
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                algorithm.parameters(), 
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 0)
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                algorithm.parameters(), 
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0),
                weight_decay=kwargs.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def create_scheduler(self, optimizer, scheduler_type: str, **kwargs):
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler ("StepLR", "ReduceLROnPlateau")
            **kwargs: Additional scheduler parameters
            
        Returns:
            Scheduler instance or None
        """
        if scheduler_type is None:
            return None
        
        if scheduler_type.lower() == "steplr":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 50),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == "reducelronplateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class DoARunner:
    """
    Main runner class for DoA estimation algorithms
    Handles both training and evaluation phases
    """
    
    def __init__(self, system_model_params, data_dict: Dict[str, Any]):
        """
        Initialize DoA Runner
        
        Args:
            system_model_params: SystemModelParams object containing all configuration
            data_dict: Dictionary containing:
                - 'measurements': Measurement data (N, T)
                - 'true_angles': True DoA angles (M,)
                - 'steering_matrix': Steering matrix (N, M)
                - 'noise': Noise data (N, T)
                - 'physical_array': Physical array positions (N,)
                - 'physical_antennas_gains': Physical antenna gains (N,)
        """
        self.system_params = system_model_params
        self.data_dict = data_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute covariance matrix once (reused across all trials)
        measurements = self.data_dict['measurements'].to(self.device).unsqueeze(0)
        self.cov_matrix = sample_covariance(measurements)
        
        # Create factory for algorithm and training components
        self.factory = AlgorithmFactory(self.device)
        
        # Create algorithm (may be recreated during window size optimization)
        self.algorithm = self.factory.create_algorithm(
            model_type=system_model_params.model_type,
            system_model_params=system_model_params
        )
        
        # Initialize training components if needed
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.results = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        if self._needs_training():
            self._setup_training()
    
    def _needs_training(self) -> bool:
        """Check if algorithm needs training"""
        return self.algorithm.training

    
    def _setup_training(self):
        """Setup training components (loss, optimizer, scheduler)"""
        # Determine loss type based on system parameters
        if self.system_params.loss_type is None:
            warnings.warn("Loss type not specified, defaulting to 'rmspe'.")
            self.system_params.loss_type = 'rmspe'
        loss_type = getattr(self.system_params, 'loss_type', 'rmspe')

        
        # Create loss function
        self.loss_fn = self.factory.create_loss_function(loss_type)
        
        # Create optimizer
        self.optimizer = self.factory.create_optimizer(
            algorithm=self.algorithm,
            optimizer_type=self.system_params.optimizer,
            learning_rate=self.system_params.learning_rate,
            weight_decay=getattr(self.system_params, 'weight_decay', 0)
        )
        
        # Create scheduler
        self.scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.system_params.scheduler,
            step_size=getattr(self.system_params, 'step_size', 50),
            patience=getattr(self.system_params, 'patience', 10)
        )
    
    def _setup_wandb(self, window_size, trial_idx=None):
        """
        Setup wandb logging if enabled
        
        Args:
            window_size: Current window size being used
            trial_idx: Trial index for optimization mode (None for single mode)
        """
        if not getattr(self.system_params, 'use_wandb', False):
            return
        
        try:
            import wandb
            
            # Create project name with model type and timestamp
            dt_string = getattr(self.system_params, 'dt_string_for_save', 
                              datetime.now().strftime("%d_%m_%Y_%H_%M"))
            project_name = f"{self.system_params.model_type}_{dt_string}"
            
            # Create run name
            if trial_idx is not None:
                run_name = f"trial_{trial_idx+1}_window_size_{window_size}_{self.system_params.loss_type}"
            else:
                run_name = f"winsow_size_{window_size}_{self.system_params.loss_type}"
            
            # Initialize wandb
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model_type": self.system_params.model_type,
                    "loss_type": self.system_params.loss_type,
                    "window_size": window_size,
                    "learning_rate": self.system_params.learning_rate,
                    "epochs": self.system_params.epochs,
                    "N": self.system_params.N,
                    "M": self.system_params.M,
                    "T": self.system_params.T,
                    "snr": self.system_params.snr,
                    "optimizer": self.system_params.optimizer,
                    "scheduler": self.system_params.scheduler,
                },
                reinit=True  # Allow multiple runs in same script
            )
            
        except ImportError:
            warnings.warn("wandb not installed. Install with 'pip install wandb' to enable logging.")
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}")
    
    def _close_wandb(self):
        """Close wandb run if active"""
        if getattr(self.system_params, 'use_wandb', False):
            try:
                import wandb
                wandb.finish()
            except:
                pass
    
    def run(self) -> Dict[str, Any]:
        """
        Main run method - handles both training and evaluation
        Supports single window size or window size optimization
        
        Returns:
            Dictionary with results including estimated DoAs and metrics
        """
        # Normalize window_size to always be a list/array for unified handling
        window_size = getattr(self.system_params, 'softmax_window_size', 21)
        if not hasattr(window_size, '__len__'):
            window_sizes = [window_size]  # Single case
            optimization_mode = False
        else:
            window_sizes = window_size  # Multiple cases
            optimization_mode = len(window_sizes) > 1
        
        if optimization_mode:
            print(f"Starting window size optimization over {len(window_sizes)} configurations...")
        else:
            print("Running single configuration...")
            
        results = self._run_with_window_optimization(window_sizes, optimization_mode)
        self.results = self.data_dict.copy()  # Copy to avoid modifying original
        self.results.update(results)  # Update with results
        

        return self.results.copy()  # Return a copy to avoid external modifications
    
    def _create_fresh_algorithm(self, window_size):
        """
        Create a new algorithm instance with fresh parameters
        
        Args:
            window_size: Window size (int, float, or relative float)
            
        Returns:
            Fresh algorithm instance with specified window size
        """
        algorithm = self.factory.create_algorithm(
            model_type=self.system_params.model_type,
            system_model_params=self.system_params
        )
        
        # Set the specific window size
        if isinstance(window_size, float) and 0 < window_size < 1:
            # Case 2: relative to grid size
            algorithm.window_size = int(window_size * len(algorithm.angles_grid))
        else:
            # Case 1: absolute size
            algorithm.window_size = int(window_size)
        
        return algorithm.to(self.device)
    
    def _run_with_window_optimization(self, window_sizes, optimization_mode=True) -> Dict[str, Any]:
        """
        Unified method for both single and multiple window size configurations
        
        Args:
            window_sizes: List of window sizes to try
            optimization_mode: Whether to print optimization-specific messages
            
        Returns:
            Results from best configuration (or single configuration) plus stats
        """

        
        best_rmspe = float('inf')
        best_results = None
        best_window_size = None
        
        
        trial_times = []
        
        for i, window_size in enumerate(window_sizes):
            trial_start_time = time.time()
            
            if optimization_mode:
                print(f"Trial {i+1}/{len(window_sizes)}: window_size={window_size}")
            else:
                print(f"Running with window_size={window_size}")
            
            # Create fresh algorithm instance with reset parameters
            self.algorithm = self._create_fresh_algorithm(window_size)
            
            # Setup fresh training components and wandb
            if self._needs_training():
                self._setup_training()
                self._setup_wandb(window_size, trial_idx=i if optimization_mode else None)
            
            # Run complete training + evaluation cycle
            results = {}
            if self._needs_training():
                train_results = self._run_training()
                results.update(train_results)
            
            eval_results = self._run_evaluation()
            results.update(eval_results)
            
            # Close wandb for this trial
            self._close_wandb()
            
            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            trial_times.append(trial_duration)
            
            # Check if this is the best configuration
            if results['rmspe'] < best_rmspe:
                best_rmspe = results['rmspe']
                best_results = results.copy()
                best_window_size = window_size
                status_msg = f"New best RMSPE: {best_rmspe:.6f}" if optimization_mode else f"RMSPE: {best_rmspe:.6f}"
                print(f"  {status_msg} (Time: {trial_duration:.2f}s)")
            else:
                print(f"  RMSPE: {results['rmspe']:.6f} (Time: {trial_duration:.2f}s)")
        
        total_time = np.sum(np.array(trial_times))
        
        if optimization_mode:
            average_duration = np.mean(trial_times)
            print(f"\nOptimization complete!")
            print(f"Best window_size: {best_window_size}, RMSPE: {best_rmspe:.6f}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per trial: {average_duration:.2f}s")
            
            # Add optimization stats
            best_results['optimal_window_size'] = best_window_size
            best_results['optimization_stats'] = {
                'total_time': total_time,
                'average_time_per_trial': average_duration,
                'trial_times': trial_times,
                'num_trials': len(window_sizes)
            }
        else:
            print(f"Completed in {total_time:.2f}s")
            best_results['execution_time'] = total_time
        
        return best_results
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run training phase using precomputed covariance matrix
        
        Returns:
            Training results
        """
        
        # Extract data
        true_angles = self.data_dict['true_angles'].to(self.device).unsqueeze(0)    # (1, M)
        M = len(self.data_dict['true_angles'])
        
        # Reset training history for fresh instance
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'steering_mse': []
        }

        print(f"Training for {self.system_params.epochs} epochs...")
        
        for epoch in tqdm(range(self.system_params.epochs), desc="Training"):
            epoch_loss = self._train_epoch(self.cov_matrix, true_angles, M)
            
            # Compute steering MSE for this epoch
            with torch.no_grad():
                steering_mse = self._compute_steering_matrix_mse(true_angles.squeeze(0))
        
            # Store training history
            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['steering_mse'].append(steering_mse)
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
            
            # Log to wandb if enabled
            if getattr(self.system_params, 'use_wandb', False):
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "steering_mse": steering_mse,  
                        "learning_rate": current_lr if self.optimizer else 0
                    })
                except:
                    pass
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.system_params.epochs}, Loss: {epoch_loss:.6f}, Steering MSE: {steering_mse:.6f}")
        
        return {
            'training_history': self.training_history,
            'final_train_loss': epoch_loss
        }
    
    def _train_epoch(self, cov_matrix: torch.Tensor, true_angles: torch.Tensor, M: int) -> float:
        """
        Train for one epoch
        
        Args:
            cov_matrix: Covariance matrix (1, N, N)
            true_angles: True angles (1, M)
            M: Number of sources
            
        Returns:
            Epoch loss
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        algorithm_result = self.algorithm(cov_matrix, M)
        if self.system_params.model_type.lower() == "diffmusic":
            estimated_angles, peaks_masks, _ , _ = algorithm_result
        if peaks_masks is None:
            warnings.warn("No peaks masks returned from algorithm, something is weird and you need to check that.")
        
        # Compute loss based on loss type
        loss_kwargs = {
            'predictions': estimated_angles,
            'targets': true_angles
        }
        
        # Add additional arguments for spectrum-based losses
        if hasattr(self.loss_fn, 'loss_type') and self.loss_fn.loss_type in ['spectrum', 'unsupervised']:
            loss_kwargs['spectrum'] = self.algorithm.music_spectrum
            loss_kwargs['angles_grid'] = self.algorithm.angles_grid
            
            if self.loss_fn.loss_type == 'unsupervised':

                loss_kwargs['peak_masks'] = peaks_masks
        
        # Compute loss
        loss = self.loss_fn(**loss_kwargs)
        
        # Handle different loss return types
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation phase using precomputed covariance matrix
        
        Returns:
            Evaluation results
        """
        
        with torch.no_grad():
            # Extract data
            true_angles = self.data_dict['true_angles'].to(self.device).unsqueeze(0)    # (1, M)
            M = len(self.data_dict['true_angles'])
            
            # Forward pass with precomputed covariance
            algorithm_result = self.algorithm(self.cov_matrix, M)
            if "music" in self.system_params.model_type.lower(): # Handle MUSIC and DiffMUSIC
                estimated_angles, _, _, _ = algorithm_result
            
            # Remove batch dimension for output
            estimated_angles = estimated_angles.squeeze(0)  # (M,)
            true_angles = true_angles.squeeze(0)  # (M,)

            learned_antenna_positions, learned_coplex_gains = self.algorithm.get_array_learnable_parameters(learnable=False)

            # Compute MSE between real and estimated steering matrices
            steering_matrix_mse = self._compute_steering_matrix_mse(true_angles)
            
            # Compute RMSPE for evaluation
            from src.metrics import RMSPELoss
            rmspe_fn = RMSPELoss()
            rmspe = rmspe_fn(estimated_angles.unsqueeze(0), true_angles.unsqueeze(0))

                   
        
        return {
            'estimated_angles': np.sort(estimated_angles.cpu().numpy()),
            'true_angles': true_angles.cpu().numpy(),
            'rmspe': rmspe.item(),
            "learned_antenna_positions": learned_antenna_positions,
            "learned_antennas_gains": learned_coplex_gains,
            'music_spectrum': self.algorithm.music_spectrum.cpu().numpy() if self.algorithm.music_spectrum is not None else None,
            'angles_grid': self.algorithm.angles_grid.cpu().numpy() if hasattr(self.algorithm, 'angles_grid') else None,
            'steering_matrix_mse': steering_matrix_mse 
        }
    
    def save_model(self, path: Path):
        """Save trained model"""
        if self._needs_training():
            torch.save({
                'model_state_dict': self.algorithm.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'system_params': self.system_params
            }, path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.algorithm.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

    def plot_graphs(self, path: Path):
        """
        Plot specified graphs by case
        """
        
        if self.results is None:
            raise ValueError("No results available. Run the algorithm first.")
        true_angles = torch.from_numpy(self.results['true_angles']).to(self.device)

        if "music" in self.system_params.model_type.lower() and self.system_params.plot_results:
                    if self.algorithm.music_spectrum is None:
                        raise ValueError("Music spectrum is None the algorithm hasn't runned yet.")
                    self.algorithm.plot_spectrum(
                        true_angles,
                        path
                    )
        # Plot learned parameters (Figure 3 style) for diffMUSIC
        if self.system_params.model_type.lower() == "diffmusic" and self.system_params.plot_results: # TODO: check with Ori
            # Get physical parameters for comparison
            physical_array = self.data_dict.get('physical_array', None)
            physical_gains = self.data_dict.get('physical_antennas_gains', None)
            
            # Convert to torch tensors if they're numpy arrays
            if physical_array is not None and not isinstance(physical_array, torch.Tensor):
                physical_array = torch.from_numpy(physical_array)
            if physical_gains is not None and not isinstance(physical_gains, torch.Tensor):
                physical_gains = torch.from_numpy(physical_gains)
            
            # Create title based on loss type and performance
            rmspe = self.results.get('rmspe', 0)
            loss_type = getattr(self.system_params, 'loss_type', 'unknown')
            title = f"Learned Parameters ({loss_type.upper()}) - RMSPE: {rmspe:.3f}°"
            
            self.algorithm.plot_learned_parameters(
                true_positions=physical_array,
                true_gains=physical_gains,
                path_to_save=path,
                title=title
            )
        return None
    

    def run_multi_loss_spectrum_comparison(self, loss_functions: List[str]) -> Dict[str, Any]:
        """
        Run the same experiment with multiple loss functions and collect spectra for comparison
        
        Args:
            loss_functions: List of loss functions to compare (e.g., ['rmspe', 'spectrum', 'unsupervised'])
            
        Returns:
            Dictionary containing spectra and results for each loss function
        """
        print(f"Starting multi-loss spectrum comparison with {len(loss_functions)} loss functions...")
        print(f"Loss functions: {loss_functions}")
        
        # Initialize results storage
        spectra_results = {}
        full_results = {}
        execution_times = {}
        
        # Get window size for consistent comparison
        window_size = getattr(self.system_params, 'softmax_window_size', 21)
        if hasattr(window_size, '__len__'):
            # If multiple window sizes, use the first one for comparison
            window_size = window_size[0] if len(window_size) > 0 else 21
        
        # Run experiment for each loss function
        for i, loss_function in enumerate(loss_functions):
            print(f"\n{'='*50}")
            print(f"Running with Loss Function: {loss_function} ({i+1}/{len(loss_functions)})")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                # Create fresh algorithm instance with reset parameters
                self.algorithm = self._create_fresh_algorithm(window_size)
                
                # Update system params for this loss function
                original_loss_type = getattr(self.system_params, 'loss_type', None)
                self.system_params.loss_type = loss_function
                
                # Setup fresh training components
                if self._needs_training():
                    self._setup_training()
                    self._setup_wandb(window_size, trial_idx=i)
                
                # Run complete training + evaluation cycle
                results = {}
                if self._needs_training():
                    train_results = self._run_training()
                    results.update(train_results)
                
                eval_results = self._run_evaluation()
                results.update(eval_results)
                
                # Store spectrum with proper dimension handling
                spectrum = results.get('music_spectrum', None)
                if spectrum is not None:
                    # Ensure consistent storage format (remove batch dimension)
                    if isinstance(spectrum, torch.Tensor):
                        spectrum = spectrum.cpu().numpy()
                    if spectrum.ndim == 2 and spectrum.shape[0] == 1:
                        spectrum = spectrum.squeeze(0)  # Remove batch dimension
                    elif spectrum.ndim == 2 and spectrum.shape[0] > 1:
                        spectrum = spectrum[0]  # Take first batch element
                    
                    spectra_results[loss_function] = spectrum
                else:
                    spectra_results[loss_function] = None
                
                full_results[loss_function] = results
                execution_times[loss_function] = time.time() - start_time
                
                print(f"  RMSPE: {results['rmspe']:.6f} (Time: {execution_times[loss_function]:.2f}s)")
                
                # Close wandb for this trial
                self._close_wandb()
                
                # Restore original loss type
                if original_loss_type is not None:
                    self.system_params.loss_type = original_loss_type
                    
            except Exception as e:
                print(f"  Error with {loss_function}: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                
                # Store error results
                spectra_results[loss_function] = None
                full_results[loss_function] = {'error': str(e), 'rmspe': float('inf')}
                execution_times[loss_function] = time.time() - start_time
        
        # Compile consolidated results
        total_time = sum(execution_times.values())
        consolidated_results = {
            'comparison_type': 'multi_loss_spectrum',
            'loss_functions': loss_functions,
            'window_size_used': window_size,
            'spectra': spectra_results,
            'results': full_results,
            'execution_times': execution_times,
            'total_execution_time': total_time,
            'shared_data': {
                'angles_grid': self.algorithm.angles_grid.cpu().numpy() if hasattr(self.algorithm, 'angles_grid') else None,
                'true_angles': self.data_dict['true_angles'].cpu().numpy() if isinstance(self.data_dict['true_angles'], torch.Tensor) else self.data_dict['true_angles'],
                'system_params': self.system_params
            }
        }
        
        print(f"\nMulti-loss spectrum comparison completed in {total_time:.2f}s")
        
        # Store results for potential later use
        self.multi_loss_results = consolidated_results
        
        return consolidated_results
        
    def plot_multi_loss_spectrum_comparison(self, multi_loss_results: Dict[str, Any], path: Path):
        """
        Plot spectra from multiple loss functions on the same figure
        
        Args:
            multi_loss_results: Results from run_multi_loss_spectrum_comparison()
            path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if multi_loss_results is None:
            raise ValueError("No multi-loss results available. Run run_multi_loss_spectrum_comparison() first.")
        
        # Extract data
        loss_functions = multi_loss_results['loss_functions']
        spectra = multi_loss_results['spectra']
        results = multi_loss_results['results']
        shared_data = multi_loss_results['shared_data']
        
        angles_grid = shared_data['angles_grid']
        true_angles = shared_data['true_angles']
        
        if angles_grid is None:
            print("Warning: No angles grid available for plotting")
            return
        
        # Convert angles to degrees for plotting
        angles_deg = np.rad2deg(angles_grid)
        true_angles_deg = np.rad2deg(true_angles)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Define colors and line styles for different loss functions FIRST
        colors = {'rmspe': 'blue', 'spectrum': 'red', 'unsupervised': 'green', 'physical': 'black'}
        line_styles = {'rmspe': '-', 'spectrum': '--', 'unsupervised': '-.', 'physical': ':'}
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        default_styles = ['-', '--', '-.', ':', '-', '--']
        
        # Compute and plot physical parameters spectrum if available
        physical_spectrum = self._compute_physical_spectrum()
        if physical_spectrum is not None:
            plt.plot(angles_deg, physical_spectrum, 
                    color=colors['physical'], linestyle=line_styles['physical'], linewidth=3,
                    label='Physical Parameters', alpha=0.8)
        
        # Plot spectrum for each loss function
        for i, loss_function in enumerate(loss_functions):
            spectrum = spectra.get(loss_function, None)
            result = results.get(loss_function, {})
            
            if spectrum is not None and 'error' not in result:
                # Handle spectrum dimensions - remove batch dimension if present
                if isinstance(spectrum, np.ndarray):
                    if spectrum.ndim == 2 and spectrum.shape[0] == 1:
                        spectrum = spectrum.squeeze(0)  # Remove batch dimension (1, 481) -> (481,)
                    elif spectrum.ndim == 2 and spectrum.shape[0] > 1:
                        spectrum = spectrum[0]  # Take first batch element
                elif isinstance(spectrum, torch.Tensor):
                    spectrum = spectrum.cpu().numpy()
                    if spectrum.ndim == 2 and spectrum.shape[0] == 1:
                        spectrum = spectrum.squeeze(0)
                    elif spectrum.ndim == 2 and spectrum.shape[0] > 1:
                        spectrum = spectrum[0]
                
                # Ensure spectrum is 1D
                if spectrum.ndim != 1:
                    print(f"Warning: Unexpected spectrum shape for {loss_function}: {spectrum.shape}")
                    continue
                    
                # Ensure angles_deg and spectrum have the same length
                if len(angles_deg) != len(spectrum):
                    print(f"Warning: Length mismatch for {loss_function}: angles={len(angles_deg)}, spectrum={len(spectrum)}")
                    continue
                
                # Get color and style
                color = colors.get(loss_function, default_colors[i % len(default_colors)])
                style = line_styles.get(loss_function, default_styles[i % len(default_styles)])
                
                # Get RMSPE for label
                rmspe = result.get('rmspe', float('inf'))
                
                # Plot spectrum
                plt.plot(angles_deg, spectrum, 
                        color=color, linestyle=style, linewidth=2,
                        label=f'{loss_function.upper()} (RMSPE: {rmspe:.3f}°)')
            else:
                print(f"Warning: No valid spectrum for {loss_function}")
        
        # Add true angle markers
        for i, true_angle in enumerate(true_angles_deg):
            plt.axvline(x=true_angle, color='black', linestyle=':', alpha=0.7, 
                    label='True DoA' if i == 0 else "")
        
        # Customize plot
        plt.xlabel('Angle [degrees]', fontsize=12, fontweight='bold')
        plt.ylabel('Spectrum Power', fontsize=12, fontweight='bold')
        plt.title('Multi-Loss Function Spectrum Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add system info as text
        steering_mse_text = ", ".join([f"{lf}: {results[lf].get('steering_matrix_mse', float('nan')):.3f}" 
                                  for lf in loss_functions if 'error' not in results.get(lf, {})])

        info_text = (f"N={shared_data['system_params'].N}, "
                    f"M={shared_data['system_params'].M}, "
                    f"T={shared_data['system_params'].T}, "
                    f"SNR={shared_data['system_params'].snr}dB\n"
                    f"Steering MSE - {steering_mse_text}")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        save_path = path / "multi_loss_spectrum_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(path / "multi_loss_spectrum_comparison.pdf", bbox_inches='tight')
        
        print(f"Multi-loss spectrum comparison plot saved to: {save_path}")
        plt.show()
        

    def plot_multi_loss_learned_parameters(self, multi_loss_results: Dict[str, Any], path: Path):
        """
        Plot learned parameters from multiple loss functions on the same figure
        
        Args:
            multi_loss_results: Results from run_multi_loss_spectrum_comparison()
            path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        if multi_loss_results is None:
            raise ValueError("No multi-loss results available. Run run_multi_loss_spectrum_comparison() first.")
        
        # Extract data
        loss_functions = multi_loss_results['loss_functions']
        results = multi_loss_results['results']
        shared_data = multi_loss_results['shared_data']
        
        # Get physical parameters for comparison
        true_positions = self.data_dict.get('physical_array', None)
        true_gains = self.data_dict.get('physical_antennas_gains', None)
        
        # Convert to numpy if they're torch tensors
        if true_positions is not None and isinstance(true_positions, torch.Tensor):
            true_positions = true_positions.cpu().numpy()
        if true_gains is not None and isinstance(true_gains, torch.Tensor):
            true_gains = true_gains.cpu().numpy()
        
        # Create figure with appropriate height for all rows
        num_loss_functions = len([lf for lf in loss_functions if 'error' not in results.get(lf, {})])
        total_rows = num_loss_functions + (1 if true_positions is not None else 0)
        fig, ax = plt.subplots(1, 1, figsize=(16, 3 + total_rows * 1.5))
        
        # Define colors for different loss functions
        colors = {'rmspe': 'blue', 'spectrum': 'red', 'unsupervised': 'green'}
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Y positions for different rows
        row_spacing = 1.5
        current_y = (total_rows - 1) * row_spacing / 2
        
        # Plot learned parameters for each loss function
        plotted_loss_functions = []
        for i, loss_function in enumerate(loss_functions):
            result = results.get(loss_function, {})
            
            if 'error' in result:
                print(f"Skipping {loss_function} due to error: {result['error']}")
                continue
            
            learned_positions = result.get('learned_antenna_positions', None)
            learned_gains = result.get('learned_antennas_gains', None)
            rmspe = result.get('rmspe', float('inf'))
            
            if learned_positions is None or learned_gains is None:
                print(f"Warning: No learned parameters for {loss_function}")
                continue
            
            # Convert positions to wavelength units for display
            wavelength = shared_data['system_params'].wavelength
            learned_positions_wl = learned_positions / (wavelength / 2)
            
            # Get color for this loss function
            color = colors.get(loss_function, default_colors[i % len(default_colors)])
            
            # Plot learned parameters
            for j, (pos, gain) in enumerate(zip(learned_positions_wl, learned_gains)):
                # Circle radius represents gain magnitude
                radius = abs(gain) * 0.2  # Smaller radius for multiple rows
                
                # Circle color and segment angle represent gain phase
                phase = np.angle(gain)
                
                # Draw circle with color corresponding to loss function
                circle = patches.Circle((pos, current_y), radius, 
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, 
                                        linewidth=2,
                                        label=f'{loss_function.upper()} (RMSPE: {rmspe:.3f}°)' if j == 0 else "")
                ax.add_patch(circle)
                
                # Draw phase segment (line from center to edge)
                segment_x = pos + radius * np.cos(phase)
                segment_y = current_y + radius * np.sin(phase)
                ax.plot([pos, segment_x], [current_y, segment_y], color=color, linewidth=2)
                
                # Add antenna number
                if j == 0:  # Only add for first antenna to avoid clutter
                    ax.text(pos - 0.3, current_y, f'{loss_function.upper()}', 
                        ha='right', va='center', fontsize=10, fontweight='bold', color=color)
            
            # Add horizontal reference line
            ax.axhline(y=current_y, color=color, linestyle=':', alpha=0.3, linewidth=1)
            
            plotted_loss_functions.append(loss_function)
            current_y -= row_spacing
        
        # Plot physical parameters if available (bottom row)
        if true_positions is not None and true_gains is not None:
            true_positions_wl = true_positions / (wavelength / 2)
            
            for j, (pos, gain) in enumerate(zip(true_positions_wl, true_gains)):
                radius = abs(gain) * 0.2
                phase = np.angle(gain)
                
                # Draw circle with different style for physical parameters
                circle = patches.Circle((pos, current_y), radius, 
                                        facecolor='lightgray', 
                                        edgecolor='black', 
                                        linewidth=2, 
                                        alpha=0.7,
                                        label='Physical' if j == 0 else "")
                ax.add_patch(circle)
                
                # Draw phase segment
                segment_x = pos + radius * np.cos(phase)
                segment_y = current_y + radius * np.sin(phase)
                ax.plot([pos, segment_x], [current_y, segment_y], 'k-', linewidth=2)
            
            # Add label for physical parameters
            ax.text(true_positions_wl[0] - 0.3, current_y, 'PHYSICAL', 
                ha='right', va='center', fontsize=10, fontweight='bold', color='black')
            
            # Add horizontal reference line
            ax.axhline(y=current_y, color='black', linestyle=':', alpha=0.3, linewidth=1)
        
        # Set axis limits and labels
        all_positions = []
        for loss_function in plotted_loss_functions:
            result = results.get(loss_function, {})
            if 'learned_antenna_positions' in result:
                positions_wl = result['learned_antenna_positions'] / (wavelength / 2)
                all_positions.extend(positions_wl)
        
        if true_positions is not None:
            all_positions.extend(true_positions_wl)
        
        if all_positions:
            ax.set_xlim(min(all_positions) - 0.5, max(all_positions) + 0.5)
        
        y_range = total_rows * row_spacing / 2 + 0.8
        ax.set_ylim(-y_range, y_range)
        ax.set_xlabel('x [λ/2]', fontsize=12, fontweight='bold')
        ax.set_ylabel('')
        
        # Create title with summary
        rmspe_summary = ', '.join([f"{lf.upper()}: {results[lf]['rmspe']:.3f}°" 
                                for lf in plotted_loss_functions])
        title = f'Multi-Loss Learned Parameters Comparison\n{rmspe_summary}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Create custom legend
        legend_elements = []
        for i, loss_function in enumerate(plotted_loss_functions):
            color = colors.get(loss_function, default_colors[i % len(default_colors)])
            rmspe = results[loss_function]['rmspe']
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                        markersize=10, markeredgecolor=color, markeredgewidth=2, 
                        alpha=0.7, label=f'{loss_function.upper()} (RMSPE: {rmspe:.3f}°)')
            )
        
        if true_positions is not None:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                        markersize=10, markeredgecolor='black', markeredgewidth=2, 
                        alpha=0.7, label='Physical')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Remove y-axis ticks for cleaner look
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save plot
        save_path = path / "multi_loss_learned_parameters.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(path / "multi_loss_learned_parameters.pdf", bbox_inches='tight')
        
        print(f"Multi-loss learned parameters plot saved to: {save_path}")
        plt.show()


    def _compute_steering_matrix_mse(self, true_angles: torch.Tensor) -> float:
        """
        Compute MSE between real and estimated steering matrices
        
        Args:
            true_angles: True DoA angles in radians
            
        Returns:
            MSE value
        """
        # Get real steering matrix from data_dict (computed during data creation)
        real_steering_matrix = self.data_dict.get('steering_matrix', None)
        
        if real_steering_matrix is None:
            return float('nan')  # Cannot compute if real steering matrix not available
        
        # Convert to torch tensor if needed and move to device
        if not isinstance(real_steering_matrix, torch.Tensor):
            real_steering_matrix = torch.from_numpy(real_steering_matrix)
        real_steering_matrix = real_steering_matrix.to(self.device)
        
        true_angles = true_angles.to(self.device)
        
        # Compute estimated steering matrix using learned parameters
        estimated_steering_matrix = self.algorithm.compute_steering_matrix(true_angles)
        
        # Compute MSE
        mse = torch.mean(torch.abs(real_steering_matrix - estimated_steering_matrix) ** 2)
        
        return mse.item()
    

    def _compute_physical_spectrum(self) -> Optional[np.ndarray]:
        """
        Compute MUSIC spectrum using physical parameters for comparison
        
        Returns:
            Physical spectrum as numpy array, or None if physical parameters not available
        """
        # Check if physical parameters are available
        physical_array = self.data_dict.get('physical_array', None)
        physical_gains = self.data_dict.get('physical_antennas_gains', None)
        
        if physical_array is None or physical_gains is None:
            print("Warning: Physical parameters not available for spectrum computation")
            return None
        
        try:
            # Convert to torch tensors if needed
            if not isinstance(physical_array, torch.Tensor):
                physical_array = torch.from_numpy(physical_array)
            if not isinstance(physical_gains, torch.Tensor):
                physical_gains = torch.from_numpy(physical_gains)
            
            # Move to device
            physical_array = physical_array.to(self.device).to(torch.float64)
            physical_gains = physical_gains.to(self.device).to(torch.complex64)
            
            # Check if noise subspace is available from the algorithm
            if not hasattr(self.algorithm, 'noise_subspace') or self.algorithm.noise_subspace is None:
                print("Warning: No noise subspace available for physical spectrum computation")
                return None
            
            # Save current learned parameters
            with torch.no_grad():
                current_positions = self.algorithm.antenna_positions.clone()
                current_gains = self.algorithm.complex_gain.clone()
                
                # Temporarily set to physical parameters
                self.algorithm.antenna_positions.copy_(physical_array)
                self.algorithm.complex_gain.copy_(physical_gains)
                
                # Compute spectrum using physical parameters
                inverse_spectrum = self.algorithm._compute_inverse_spectrum(self.algorithm.noise_subspace)
                physical_spectrum = 1 / (inverse_spectrum + 1e-10)
                
                # Remove batch dimension and convert to numpy
                if physical_spectrum.dim() == 2 and physical_spectrum.shape[0] == 1:
                    physical_spectrum = physical_spectrum.squeeze(0)
                physical_spectrum = physical_spectrum.cpu().detach().numpy()
                
                # Restore learned parameters
                self.algorithm.antenna_positions.copy_(current_positions)
                self.algorithm.complex_gain.copy_(current_gains)
            
            return physical_spectrum
            
        except Exception as e:
            print(f"Error computing physical spectrum: {e}")
            import traceback
            traceback.print_exc()  # This will help debug any remaining issues
            return None