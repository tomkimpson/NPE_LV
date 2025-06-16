"""
Utility functions for NPE-LV project.
"""
import numpy as np
import torch
from typing import Tuple, Optional
from sbi.utils import BoxUniform


def create_lv_prior() -> BoxUniform:
    """
    Create prior distribution for LV parameters.
    
    Returns:
    --------
    prior : BoxUniform
        Prior over [alpha, beta, delta, gamma]
    """
    low = torch.tensor([0.01, 0.001, 0.001, 0.01])   # [alpha, beta, delta, gamma]
    high = torch.tensor([1.0, 0.1, 0.1, 1.0])
    
    return BoxUniform(low=low, high=high)


def create_time_grid(t_max: float = 10.0, dt: float = 0.1) -> np.ndarray:
    """
    Create regular time grid for interpolation.
    
    Parameters:
    -----------
    t_max : float
        Maximum time
    dt : float
        Time step
        
    Returns:
    --------
    t_grid : np.ndarray
        Regular time grid
    """
    return np.arange(0, t_max + dt, dt)


def flatten_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    Flatten trajectory for neural network input.
    
    Parameters:
    -----------
    trajectory : np.ndarray of shape (n_time, 2)
        Time series of [prey, predator]
        
    Returns:
    --------
    flattened : np.ndarray of shape (n_time * 2,)
        Flattened trajectory
    """
    return trajectory.flatten()


def compute_summary_stats(trajectory: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics from trajectory.
    
    Parameters:
    -----------
    trajectory : np.ndarray of shape (n_time, 2)
        Population trajectory
    t_grid : np.ndarray
        Time points
        
    Returns:
    --------
    stats : np.ndarray
        Summary statistics vector
    """
    prey, predator = trajectory[:, 0], trajectory[:, 1]
    
    stats = []
    
    for pop in [prey, predator]:
        # Basic statistics
        stats.extend([
            np.mean(pop),
            np.std(pop),
            np.min(pop), 
            np.max(pop)
        ])
        
        # Temporal features
        if len(pop) > 1:
            # Time of peak
            peak_idx = np.argmax(pop)
            stats.append(t_grid[peak_idx])
            
            # Simple autocorrelation at lag 1
            if len(pop) > 2:
                autocorr = np.corrcoef(pop[:-1], pop[1:])[0, 1]
                stats.append(autocorr if not np.isnan(autocorr) else 0.0)
            else:
                stats.append(0.0)
        else:
            stats.extend([0.0, 0.0])
    
    return np.array(stats)


def check_trajectory_validity(trajectory: np.ndarray) -> bool:
    """
    Check if trajectory is valid (no NaN, reasonable values).
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Population trajectory
        
    Returns:
    --------
    is_valid : bool
        Whether trajectory is valid
    """
    if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
        return False
    
    if np.any(trajectory < 0):
        return False
        
    # Check for reasonable population sizes (not too large)
    if np.any(trajectory > 1e6):
        return False
        
    return True


def normalize_trajectory(trajectory: np.ndarray, 
                        method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Normalize trajectory for neural network training.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Input trajectory
    method : str
        Normalization method ('minmax' or 'zscore')
        
    Returns:
    --------
    normalized : np.ndarray
        Normalized trajectory
    stats : dict
        Normalization statistics for inverse transform
    """
    if method == 'minmax':
        min_val = np.min(trajectory, axis=0, keepdims=True)
        max_val = np.max(trajectory, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        normalized = (trajectory - min_val) / range_val
        stats = {'min': min_val, 'max': max_val}
        
    elif method == 'zscore':
        mean_val = np.mean(trajectory, axis=0, keepdims=True)
        std_val = np.std(trajectory, axis=0, keepdims=True)
        
        # Avoid division by zero
        std_val[std_val == 0] = 1.0
        
        normalized = (trajectory - mean_val) / std_val
        stats = {'mean': mean_val, 'std': std_val}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats