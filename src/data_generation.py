"""
Data generation for NPE training on Lotka-Volterra model.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

from .simulator import simulate_lv_batch
from .utils import create_lv_prior, create_time_grid, flatten_trajectory, check_trajectory_validity


class LVDataGenerator:
    """Data generator for Lotka-Volterra NPE training."""
    
    def __init__(self, 
                 x0: Tuple[int, int] = (50, 100),
                 t_max: float = 10.0,
                 dt: float = 0.1,
                 use_summary_stats: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize data generator.
        
        Parameters:
        -----------
        x0 : tuple of int
            Initial conditions (prey, predator)
        t_max : float
            Maximum simulation time
        dt : float
            Time step for interpolation grid
        use_summary_stats : bool
            Whether to use summary statistics instead of full trajectory
        seed : int, optional
            Random seed for reproducibility
        """
        self.x0 = x0
        self.t_max = t_max
        self.dt = dt
        self.use_summary_stats = use_summary_stats
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.prior = create_lv_prior()
        self.t_grid = create_time_grid(t_max, dt)
        
        # Statistics for tracking data quality
        self.failed_simulations = 0
        self.total_simulations = 0
        
    def generate_single(self, theta: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate single (theta, x) pair.
        
        Parameters:
        -----------
        theta : np.ndarray, optional
            Parameter vector. If None, sample from prior.
            
        Returns:
        --------
        theta_tensor : torch.Tensor
            Parameter vector
        x_tensor : torch.Tensor  
            Observation (trajectory or summary stats)
        """
        if theta is None:
            theta = self.prior.sample((1,)).numpy()[0]
        
        # Simulate trajectory
        from .simulator import gillespie_lv
        
        try:
            _, trajectory = gillespie_lv(
                theta=theta,
                x0=self.x0,
                t_max=self.t_max,
                t_grid=self.t_grid,
                max_steps=100000
            )
            
            # Check validity
            if not check_trajectory_validity(trajectory):
                raise ValueError("Invalid trajectory")
                
            # Convert to observation
            if self.use_summary_stats:
                from .utils import compute_summary_stats
                x = compute_summary_stats(trajectory, self.t_grid)
            else:
                x = flatten_trajectory(trajectory)
                
            return torch.tensor(theta, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
            
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            self.failed_simulations += 1
            # Return placeholder data
            if self.use_summary_stats:
                x_dim = 12  # Based on compute_summary_stats
            else:
                x_dim = len(self.t_grid) * 2
            return (torch.tensor(theta, dtype=torch.float32), 
                   torch.zeros(x_dim, dtype=torch.float32))
    
    def generate_batch(self, n_samples: int, batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch of training data.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        batch_size : int
            Size of batches for processing
            
        Returns:
        --------
        theta_batch : torch.Tensor of shape (n_samples, 4)
            Parameter vectors
        x_batch : torch.Tensor of shape (n_samples, x_dim)
            Observations
        """
        theta_list = []
        x_list = []
        
        self.total_simulations += n_samples
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Generating data"):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Sample parameters
            theta_batch = self.prior.sample((current_batch_size,)).numpy()
            
            # Simulate trajectories
            try:
                trajectories = simulate_lv_batch(
                    theta_batch=theta_batch,
                    x0=self.x0,
                    t_max=self.t_max,
                    t_grid=self.t_grid
                )
                
                # Process each trajectory
                for j in range(current_batch_size):
                    trajectory = trajectories[j]
                    
                    if check_trajectory_validity(trajectory):
                        if self.use_summary_stats:
                            from .utils import compute_summary_stats
                            x = compute_summary_stats(trajectory, self.t_grid)
                        else:
                            x = flatten_trajectory(trajectory)
                            
                        theta_list.append(theta_batch[j])
                        x_list.append(x)
                    else:
                        self.failed_simulations += 1
                        
            except Exception as e:
                warnings.warn(f"Batch simulation failed: {e}")
                self.failed_simulations += current_batch_size
        
        if len(theta_list) == 0:
            raise RuntimeError("No valid simulations generated")
            
        theta_tensor = torch.tensor(np.array(theta_list), dtype=torch.float32)
        x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
        
        print(f"Generated {len(theta_list)}/{n_samples} valid samples "
              f"({self.failed_simulations} failed)")
        
        return theta_tensor, x_tensor
    
    def save_data(self, 
                  theta: torch.Tensor, 
                  x: torch.Tensor, 
                  filepath: str,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Save generated data to file.
        
        Parameters:
        -----------
        theta : torch.Tensor
            Parameter vectors
        x : torch.Tensor
            Observations
        filepath : str
            Output filepath
        metadata : dict, optional
            Additional metadata to save
        """
        data = {
            'theta': theta,
            'x': x,
            'metadata': {
                'x0': self.x0,
                't_max': self.t_max,
                'dt': self.dt,
                'use_summary_stats': self.use_summary_stats,
                'n_samples': len(theta),
                'failed_simulations': self.failed_simulations,
                'total_simulations': self.total_simulations,
                **(metadata or {})
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved data to {filepath}")
    
    @staticmethod
    def load_data(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load data from file.
        
        Parameters:
        -----------
        filepath : str
            Input filepath
            
        Returns:
        --------
        theta : torch.Tensor
            Parameter vectors
        x : torch.Tensor
            Observations  
        metadata : dict
            Metadata
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        return data['theta'], data['x'], data['metadata']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'total_simulations': self.total_simulations,
            'failed_simulations': self.failed_simulations,
            'success_rate': 1.0 - (self.failed_simulations / max(1, self.total_simulations))
        }