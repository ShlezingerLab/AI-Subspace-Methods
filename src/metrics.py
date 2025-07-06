import torch
import torch.nn as nn
import numpy as np
from itertools import permutations


class RMSPELoss(nn.Module):
    """
    Root Mean Squared Periodic Error (RMSPE) loss for angle estimation
    Handles the periodic nature of angles and permutation invariance
    """
    
    def __init__(self, balance_factor: float = None):
        """
        Args:
            balance_factor: Weighting factor for the loss (1.0 for getting the loss as is,
            sqrt(M) is the deafult value if not speccified - by the definition in the paper)
        """
        super(RMSPELoss, self).__init__()
        self.balance_factor = balance_factor
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute RMSPE loss between predicted and target angles
        
        Args:
            predictions: Predicted angles in radians, shape (batch_size, M)
            targets: Target angles in radians, shape (batch_size, M)
            
        Returns:
            Loss tensor, shape (batch_size,)
        """
        batch_size, M = predictions.shape
        
        if M == 1:
            # Single source case - no permutation needed
            return self._periodic_mse(predictions, targets)
        
        # Multi-source case - find best permutation
        min_loss = torch.full((batch_size,), float('inf'), device=predictions.device)
        
        for perm in permutations(range(M)):
            perm_tensor = torch.tensor(perm, device=predictions.device)
            perm_predictions = predictions[:, perm_tensor]
            loss = self._periodic_mse(perm_predictions, targets)
            min_loss = torch.min(min_loss, loss)
        
        if self.balance_factor is None:
            # Default balance factor is sqrt(M) as per the paper
            self.balance_factor = 1/torch.sqrt(torch.tensor(M, dtype=torch.float32, device=predictions.device))

        return min_loss * self.balance_factor 
    
    def _periodic_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute periodic MSE for angles
        
        Args:
            pred: Predicted angles, shape (batch_size, M)
            target: Target angles, shape (batch_size, M)
            
        Returns:
            MSE loss per batch, shape (batch_size,)
        """
        # Compute angular difference considering periodicity
        diff = pred - target
        # Wrap to [-π, π] (not critical for angles in [-π/2, π/2] but here for generality)
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # Compute MSE
        mse = torch.mean(diff ** 2, dim=1)
        
        return torch.sqrt(mse)


class SpectrumLoss(nn.Module):
    """
    Spectrum-based loss (LSL,P from paper)
    Maximizes spectrum amplitude at true DoA locations
    """
    
    def __init__(self):
        super(SpectrumLoss, self).__init__()
    
    def forward(self, spectrum: torch.Tensor, true_angles: torch.Tensor, 
                angles_grid: torch.Tensor) -> torch.Tensor:
        """
        Compute spectrum loss by maximizing amplitude at true DoA locations
        
        Args:
            spectrum: MUSIC spectrum, shape (batch_size, num_angles)
            true_angles: True DoAs, shape (batch_size, M)
            angles_grid: Angular grid, shape (num_angles,)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = spectrum.shape[0]
        total_loss = 0
        
        for batch in range(batch_size):
            for angle in true_angles[batch]:
                # Find closest angle in grid
                angle_idx = torch.argmin(torch.abs(angles_grid - angle))
                # Negative spectrum value (to maximize)
                total_loss -= spectrum[batch, angle_idx]
        
        # Uncomment the following row to normalize also by the number of true angles, this is not done
        # here for consistency with the paper loss definition. Doesn't influence the optimization.
        # total_loss /= true_angles.shape[1]
        return total_loss / batch_size


class JainsIndexLoss(nn.Module):
    """
    Jain's Index loss for unsupervised learning
    Encourages sharp, concentrated peaks in the spectrum
    """
    
    def __init__(self):
        super(JainsIndexLoss, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jain's Index: J(x) = (sum(x))^2 / (n * sum(x^2))
        
        Args:
            x: Input tensor (spectrum values)
            
        Returns:
            Jain's index value (higher = more concentrated)
        """
        n = x.shape[0]
        sum_x = torch.sum(x)
        sum_x_squared = torch.sum(x ** 2)
        
        jains_index = (sum_x ** 2) / (n * sum_x_squared + 1e-8)
        
        # Return negative to minimize (we want to maximize Jain's index)
        return -jains_index


class UnsupervisedSpectrumLoss(nn.Module):
    """
    Unsupervised loss using Jain's Index on spectrum peaks (LUL from paper)
    """
    
    def __init__(self):
        super(UnsupervisedSpectrumLoss, self).__init__()
        self.jains_index_loss = JainsIndexLoss()
    
    def forward(self, spectrum: torch.Tensor, peak_masks: list) -> torch.Tensor:
        """
        Compute unsupervised loss using Jain's Index on spectrum peaks
        
        Args:
            spectrum: MUSIC spectrum, shape (batch_size, num_angles)
            peak_masks: List of masks for each peak region
            
        Returns:
            Loss value encouraging sharp peaks
        """
        total_loss = 0
        batch_size = spectrum.shape[0]
        
        for batch in range(batch_size):
            spectrum_batch = spectrum[batch]
            
            # Apply Jain's index to each peak region
            for mask_indices in peak_masks[batch]:
                masked_spectrum = spectrum_batch[mask_indices]
                
                # Jain's index encourages sharp peaks
                jains_loss = self.jains_index_loss(masked_spectrum)
                #Uncomment the following line to normalize by the number of peaks
                # jains_loss /= len(peak_masks[batch])  # Normalize by number of peaks/
                total_loss += jains_loss
        
        return total_loss / batch_size