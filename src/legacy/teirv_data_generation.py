"""
Data generation for NPE training on TEIRV viral dynamics model.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

from .teirv_simulator import simulate_teirv_batch, check_teirv_trajectory_validity
from .teirv_utils import (create_teirv_prior, get_teirv_initial_conditions, 
                        create_teirv_time_grid, apply_observation_model)


class TEIRVDataGenerator:
    """Data generator for TEIRV NPE training."""
    
    def __init__(self, 
                 t_max: float = 14.0,
                 dt: float = 1.0,
                 observation_noise: float = 1.0,
                 detection_limit: float = -0.65,
                 use_observations_only: bool = True,
                 seed: Optional[int] = None):
        """
        Initialize TEIRV data generator.
        
        Parameters:
        -----------
        t_max : float
            Maximum simulation time (days)
        dt : float
            Time step for observations (days) 
        observation_noise : float
            Standard deviation of RT-PCR observation noise
        detection_limit : float
            RT-PCR detection limit (log₁₀ scale)
        use_observations_only : bool
            If True, only return RT-PCR observations (realistic)
            If False, return full state trajectories (for debugging)
        seed : int, optional
            Random seed for reproducibility
        """
        self.t_max = t_max
        self.dt = dt
        self.observation_noise = observation_noise
        self.detection_limit = detection_limit
        self.use_observations_only = use_observations_only
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.prior = create_teirv_prior()
        self.t_grid = create_teirv_time_grid(t_max, dt)
        self.base_ic = get_teirv_initial_conditions()
        
        # Statistics for tracking data quality
        self.failed_simulations = 0
        self.total_simulations = 0
        
    def generate_single(self, theta: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate single (theta, x) pair.
        
        Parameters:
        -----------
        theta : np.ndarray, optional
            Parameter vector [β, π, δ, φ, ρ, V₀]. If None, sample from prior.
            
        Returns:
        --------
        theta_tensor : torch.Tensor
            Parameter vector
        x_tensor : torch.Tensor  
            Observation (RT-PCR time series or full trajectory)
        """
        if theta is None:
            theta = self.prior.sample().numpy()
        
        # Set up initial conditions with V₀ from parameters
        ic = self.base_ic.copy()
        ic['V'] = theta[5]  # V₀ is last parameter
        
        # Simulate trajectory
        from teirv_simulator import gillespie_teirv
        
        try:
            _, trajectory = gillespie_teirv(
                theta=theta,
                initial_conditions=ic,
                t_max=self.t_max,
                t_grid=self.t_grid,
                max_steps=100000  # Reduced from 1M to 100K for performance
            )
            
            # Check validity
            if not check_teirv_trajectory_validity(trajectory):
                raise ValueError("Invalid trajectory")
                
            # Convert to observation
            if self.use_observations_only:
                # Apply RT-PCR observation model to viral load (V compartment)
                V_trajectory = trajectory[:, 4]  # V is 5th compartment
                observations = apply_observation_model(
                    V_trajectory=V_trajectory,
                    sigma=self.observation_noise,
                    detection_limit=self.detection_limit,
                    add_noise=True
                )
                x = observations
            else:
                # Use full trajectory (flatten all compartments)
                x = trajectory.flatten()
                
            return torch.tensor(theta, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
            
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            self.failed_simulations += 1
            
            # Return placeholder data
            if self.use_observations_only:
                x_dim = len(self.t_grid)  # RT-PCR time series length
                placeholder_obs = np.full(x_dim, self.detection_limit)  # All below detection
            else:
                x_dim = len(self.t_grid) * 5  # Full trajectory (5 compartments)
                placeholder_obs = np.zeros(x_dim)
                
            return (torch.tensor(theta, dtype=torch.float32), 
                   torch.tensor(placeholder_obs, dtype=torch.float32))
    
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
        theta_batch : torch.Tensor of shape (n_samples, 6)
            Parameter vectors [β, π, δ, φ, ρ, V₀]
        x_batch : torch.Tensor of shape (n_samples, x_dim)
            Observations (RT-PCR or full trajectory)
        """
        theta_list = []
        x_list = []
        
        self.total_simulations += n_samples
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Generating TEIRV data"):
            current_batch_size = min(batch_size, n_samples - i)
            
            # Sample parameters
            theta_batch = self.prior.sample((current_batch_size,)).numpy()
            
            # Set up initial conditions for batch
            ic_batch = []
            for j in range(current_batch_size):
                ic = self.base_ic.copy()
                ic['V'] = theta_batch[j, 5]  # Set V₀ from parameter
                ic_batch.append(ic)
            
            # Simulate trajectories
            try:
                # Use base initial conditions (V₀ will be set in simulate_teirv_batch)
                trajectories = simulate_teirv_batch(
                    theta_batch=theta_batch,
                    initial_conditions=self.base_ic,
                    t_max=self.t_max,
                    t_grid=self.t_grid
                )
                
                # Process each trajectory
                for j in range(current_batch_size):
                    trajectory = trajectories[j]
                    
                    if check_teirv_trajectory_validity(trajectory):
                        # Convert to observation format
                        if self.use_observations_only:
                            # Apply observation model to viral load
                            V_trajectory = trajectory[:, 4]
                            observations = apply_observation_model(
                                V_trajectory=V_trajectory,
                                sigma=self.observation_noise,
                                detection_limit=self.detection_limit,
                                add_noise=True
                            )
                            x = observations
                        else:
                            # Use full trajectory
                            x = trajectory.flatten()
                            
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
                't_max': self.t_max,
                'dt': self.dt,
                'observation_noise': self.observation_noise,
                'detection_limit': self.detection_limit,
                'use_observations_only': self.use_observations_only,
                'n_samples': len(theta),
                'failed_simulations': self.failed_simulations,
                'total_simulations': self.total_simulations,
                'observation_dim': x.shape[1],
                'parameter_dim': theta.shape[1],
                **(metadata or {})
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved TEIRV data to {filepath}")
        print(f"Data shapes: theta={theta.shape}, x={x.shape}")
    
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
            'success_rate': 1.0 - (self.failed_simulations / max(1, self.total_simulations)),
            'observation_type': 'RT-PCR only' if self.use_observations_only else 'Full trajectory',
            'time_points': len(self.t_grid),
            'observation_noise': self.observation_noise,
            'detection_limit': self.detection_limit
        }
    
    def generate_test_patient_data(self, 
                                   true_theta: np.ndarray,
                                   noise_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic patient data for testing NPE.
        
        Parameters:
        -----------
        true_theta : np.ndarray
            True parameter values [β, π, δ, φ, ρ, V₀]
        noise_seed : int, optional
            Seed for observation noise
            
        Returns:
        --------
        times : np.ndarray
            Time points (days)
        observations : np.ndarray
            RT-PCR observations (log₁₀ scale)
        """
        # Set up initial conditions
        ic = self.base_ic.copy()
        ic['V'] = true_theta[5]
        
        # Simulate trajectory
        from teirv_simulator import gillespie_teirv
        
        _, trajectory = gillespie_teirv(
            theta=true_theta,
            initial_conditions=ic,
            t_max=self.t_max,
            t_grid=self.t_grid
        )
        
        # Apply observation model
        V_trajectory = trajectory[:, 4]
        observations = apply_observation_model(
            V_trajectory=V_trajectory,
            sigma=self.observation_noise,
            detection_limit=self.detection_limit,
            add_noise=True,
            seed=noise_seed
        )
        
        return self.t_grid, observations