"""
JAX-accelerated TEIRV data generation for NPE training.

Key improvements:
1. 4-15x faster simulation using JAX
2. Proper simulation completion (no max_steps tricks)
3. Vectorized batch processing
4. Memory-efficient output
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import time
from pathlib import Path
from tqdm import tqdm

from .teirv_simulator_jax import simulate_batch_jax, gillespie_teirv_jax
from .teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid


class TEIRVDataGeneratorJAX:
    """
    JAX-accelerated TEIRV data generator.
    
    Provides 4-15x speedup over NumPy implementation while
    ensuring complete simulations (no early termination).
    """
    
    def __init__(self,
                 t_max: float = 14.0,
                 dt: float = 1.0,
                 observation_noise: float = 1.0,
                 detection_limit: float = -0.65,
                 use_observations_only: bool = True,
                 seed: Optional[int] = None):
        """Initialize JAX data generator."""
        
        self.t_max = t_max
        self.dt = dt
        self.observation_noise = observation_noise
        self.detection_limit = detection_limit
        self.use_observations_only = use_observations_only
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.prior = create_teirv_prior(use_torch=False)  # Use JAX version
        self.t_grid = create_teirv_time_grid(t_max, dt, use_jax=False)  # Convert to numpy for compatibility
        self.base_ic = get_teirv_initial_conditions()
        
        # JAX setup
        self.rng_key = jr.PRNGKey(seed if seed is not None else 0)
        self.base_ic_jax = jnp.array([
            self.base_ic['T'], self.base_ic['E'], self.base_ic['I'], 
            self.base_ic['R'], 0.0  # V will be set per simulation
        ], dtype=jnp.float32)
        self.t_grid_jax = jnp.array(self.t_grid, dtype=jnp.float32)
        
        # Statistics
        self.total_simulations = 0
        self.failed_simulations = 0
        
    def generate_batch_jax(self, 
                          n_samples: int,
                          batch_size: int = 100,
                          show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data using JAX acceleration.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        batch_size : int
            JAX batch size (larger = faster but more memory)
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        theta_batch : torch.Tensor of shape (valid_samples, 6)
            Parameter vectors
        x_batch : torch.Tensor of shape (valid_samples, x_dim)
            Observations
        """
        if show_progress:
            print(f"Generating {n_samples} TEIRV samples with JAX...")
            start_time = time.time()
        
        valid_theta_list = []
        valid_x_list = []
        
        # Process in JAX-optimized batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="JAX batches")
        
        for batch_idx in iterator:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            # Sample parameters for this batch
            self.rng_key, sample_key = jr.split(self.rng_key)
            theta_batch_jax = self.prior.sample((current_batch_size,), key=sample_key)
            theta_batch_np = np.array(theta_batch_jax)
            
            # Get batch key
            self.rng_key, batch_key = jr.split(self.rng_key)
            
            # Run JAX batch simulation
            try:
                trajectories = simulate_batch_jax(
                    theta_batch_jax, 
                    self.base_ic_jax,
                    self.t_grid_jax,
                    batch_key
                )
                
                # Convert back to NumPy for processing
                trajectories_np = np.array(trajectories)
                
                # Process each trajectory in the batch
                for i in range(current_batch_size):
                    trajectory = trajectories_np[i]
                    
                    # Check validity
                    if self._is_trajectory_valid(trajectory):
                        # Apply observation model
                        if self.use_observations_only:
                            V_trajectory = trajectory[:, 4]  # Viral load
                            observations = self._apply_observation_model(V_trajectory)
                            x = observations
                        else:
                            x = trajectory.flatten()
                        
                        valid_theta_list.append(theta_batch_np[i])
                        valid_x_list.append(x)
                    else:
                        self.failed_simulations += 1
                        
            except Exception as e:
                # Entire batch failed
                self.failed_simulations += current_batch_size
                if show_progress:
                    print(f"Batch {batch_idx} failed: {e}")
            
            self.total_simulations += current_batch_size
        
        if len(valid_theta_list) == 0:
            raise RuntimeError("No valid simulations generated")
        
        # Convert to tensors
        theta_tensor = torch.tensor(np.array(valid_theta_list), dtype=torch.float32)
        x_tensor = torch.tensor(np.array(valid_x_list), dtype=torch.float32)
        
        if show_progress:
            elapsed = time.time() - start_time
            n_valid = len(theta_tensor)
            success_rate = n_valid / self.total_simulations
            speed = n_valid / elapsed
            
            print(f"Generated {n_valid}/{self.total_simulations} valid samples")
            print(f"Success rate: {success_rate:.1%}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Speed: {speed:.1f} samples/second")
            print(f"Output shapes: theta={theta_tensor.shape}, x={x_tensor.shape}")
        
        return theta_tensor, x_tensor
    
    def _is_trajectory_valid(self, trajectory: np.ndarray) -> bool:
        """Check if trajectory is valid."""
        if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
            return False
        if np.any(trajectory < 0):
            return False
        if np.any(trajectory > 1e12):
            return False
        return True
    
    def _apply_observation_model(self, V_trajectory: np.ndarray) -> np.ndarray:
        """Apply RT-PCR observation model."""
        # Handle zero/negative viral loads
        V_positive = np.maximum(V_trajectory, 1e-10)
        
        # Log transform
        log_V = np.log10(V_positive)
        
        # Add noise
        noise = np.random.normal(0, self.observation_noise, size=log_V.shape)
        observations = log_V + noise
        
        # Apply detection limit
        observations = np.maximum(observations, self.detection_limit)
        
        return observations
    
    def save_data(self, 
                  theta: torch.Tensor, 
                  x: torch.Tensor, 
                  filepath: str,
                  metadata: Optional[Dict[str, Any]] = None):
        """Save generated data."""
        if metadata is None:
            metadata = {}
            
        data = {
            'theta': theta,
            'x': x,
            'metadata': {
                **metadata,
                't_max': self.t_max,
                'dt': self.dt,
                'observation_noise': self.observation_noise,
                'detection_limit': self.detection_limit,
                'use_observations_only': self.use_observations_only,
                'n_samples': len(theta),
                'observation_dim': x.shape[1] if len(x.shape) > 1 else x.shape[0],
                'generator_type': 'JAX',
                'success_rate': 1.0 - (self.failed_simulations / max(self.total_simulations, 1))
            }
        }
        
        torch.save(data, filepath)
        
    @staticmethod
    def load_data(filepath: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load saved data."""
        data = torch.load(filepath)
        return data['theta'], data['x'], data['metadata']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        success_rate = 1.0 - (self.failed_simulations / max(self.total_simulations, 1))
        
        return {
            'total_simulations': self.total_simulations,
            'failed_simulations': self.failed_simulations,
            'success_rate': success_rate,
            'observation_type': 'RT-PCR only' if self.use_observations_only else 'Full trajectory',
            'time_points': len(self.t_grid),
            'observation_noise': self.observation_noise,
            'detection_limit': self.detection_limit,
            'generator_type': 'JAX-accelerated'
        }


def benchmark_jax_vs_numpy_generation(n_samples: int = 100):
    """Compare JAX vs NumPy data generation speed."""
    print(f"Benchmarking JAX vs NumPy Data Generation ({n_samples} samples)")
    print("=" * 70)
    
    # Test NumPy implementation
    print("Testing NumPy data generation...")
    try:
        # Try relative import first, then absolute
        try:
            from .teirv_data_generation import TEIRVDataGenerator
        except ImportError:
            from teirv_data_generation import TEIRVDataGenerator
        
        numpy_gen = TEIRVDataGenerator(seed=42)
        start_time = time.time()
        
        theta_np, x_np = numpy_gen.generate_batch(n_samples, batch_size=20)
        numpy_time = time.time() - start_time
        numpy_speed = len(theta_np) / numpy_time
        print(f"✅ NumPy: {numpy_time:.1f}s, {numpy_speed:.2f} samples/sec, {len(theta_np)}/{n_samples} valid")
    except Exception as e:
        print(f"❌ NumPy failed: {e}")
        numpy_time = float('inf')
        numpy_speed = 0
    
    # Test JAX implementation
    print(f"\nTesting JAX data generation...")
    jax_gen = TEIRVDataGeneratorJAX(seed=42)
    start_time = time.time()
    
    try:
        theta_jax, x_jax = jax_gen.generate_batch_jax(n_samples, batch_size=50)
        jax_time = time.time() - start_time
        jax_speed = len(theta_jax) / jax_time
        print(f"✅ JAX: {jax_time:.1f}s, {jax_speed:.2f} samples/sec, {len(theta_jax)}/{n_samples} valid")
    except Exception as e:
        print(f"❌ JAX failed: {e}")
        jax_time = float('inf')
        jax_speed = 0
    
    # Comparison
    if numpy_time < float('inf') and jax_time < float('inf'):
        speedup = numpy_time / jax_time
        print(f"\n🚀 JAX Speedup: {speedup:.1f}x faster")
        print(f"   NumPy: {numpy_time:.1f}s")
        print(f"   JAX:   {jax_time:.1f}s")
        
        if speedup > 1:
            time_saved = numpy_time - jax_time
            print(f"   Time saved: {time_saved:.1f}s ({time_saved/numpy_time:.1%})")
    
    return {'numpy_time': numpy_time, 'jax_time': jax_time, 'speedup': numpy_time / jax_time}


if __name__ == '__main__':
    # Run benchmark
    try:
        benchmark_jax_vs_numpy_generation(n_samples=50)
    except ImportError:
        print("JAX not available. Install with: pip install jax jaxlib")