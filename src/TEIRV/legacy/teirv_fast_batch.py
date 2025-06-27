"""
Fast batch processing for TEIRV - optimized sequential approach.

Instead of complex multiprocessing, this focuses on eliminating
the overhead in the sequential batch processing loop.
"""
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional, List
import time
import warnings
from tqdm import tqdm

from .teirv_simulator import gillespie_teirv, check_teirv_trajectory_validity
from .teirv_utils import apply_observation_model, create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid


def simulate_teirv_batch_fast(
    theta_batch: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast sequential batch simulation with minimal overhead.
    
    Key optimizations:
    1. Eliminate repeated dict copying 
    2. Pre-allocate all arrays
    3. Minimize function call overhead
    4. Efficient validity checking
    
    Parameters:
    -----------
    theta_batch : np.ndarray of shape (n_batch, 6)
        Parameter vectors [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Base initial conditions
    t_max : float
        Maximum simulation time
    t_grid : np.ndarray
        Time grid for interpolation
    **kwargs : additional arguments for gillespie_teirv
        
    Returns:
    --------
    trajectories : np.ndarray of shape (n_batch, len(t_grid), 5)
        Batch of trajectories
    success_mask : np.ndarray of shape (n_batch,)
        Boolean mask indicating successful simulations
    """
    n_batch = theta_batch.shape[0]
    n_time = len(t_grid)
    
    # Pre-allocate results
    trajectories = np.zeros((n_batch, n_time, 5))
    success_mask = np.zeros(n_batch, dtype=bool)
    
    # Extract base initial conditions once
    base_T = initial_conditions['T']
    base_E = initial_conditions['E'] 
    base_I = initial_conditions['I']
    base_R = initial_conditions['R']
    
    # Process each simulation with minimal overhead
    for i in range(n_batch):
        try:
            # Create initial conditions efficiently (avoid dict operations)
            ic = {
                'T': base_T,
                'E': base_E,
                'I': base_I, 
                'R': base_R,
                'V': theta_batch[i, 5]  # V0 from parameters
            }
            
            # Simulate
            _, traj = gillespie_teirv(
                theta=theta_batch[i],
                initial_conditions=ic,
                t_max=t_max,
                t_grid=t_grid,
                **kwargs
            )
            
            # Fast validity check (inline for speed)
            if not (np.any(np.isnan(traj)) or np.any(np.isinf(traj)) or 
                   np.any(traj < 0) or np.any(traj > 1e12)):
                trajectories[i] = traj
                success_mask[i] = True
            
        except Exception:
            # Silently continue on failure (trajectory stays zeros)
            pass
    
    return trajectories, success_mask


def apply_observation_model_batch(
    V_trajectories: np.ndarray,
    observation_noise: float = 1.0,
    detection_limit: float = -0.65
) -> np.ndarray:
    """
    Apply RT-PCR observation model to batch of viral trajectories.
    
    Vectorized implementation for maximum speed.
    
    Parameters:
    -----------
    V_trajectories : np.ndarray of shape (n_batch, n_time)
        Viral load trajectories
    observation_noise : float
        RT-PCR noise standard deviation
    detection_limit : float
        Detection threshold
        
    Returns:
    --------
    observations : np.ndarray of shape (n_batch, n_time)
        RT-PCR observations
    """
    # Vectorized operations for entire batch
    V_positive = np.maximum(V_trajectories, 1e-10)
    log_V = np.log10(V_positive)
    
    # Add noise to entire batch at once
    noise = np.random.normal(0, observation_noise, size=log_V.shape)
    observations = log_V + noise
    
    # Apply detection limit
    observations = np.maximum(observations, detection_limit)
    
    return observations


class FastTEIRVDataGenerator:
    """
    Fast TEIRV data generator using optimized sequential processing.
    
    This approach focuses on eliminating overhead in the sequential loop
    rather than complex parallelization, which can be more effective
    for moderate batch sizes.
    """
    
    def __init__(self,
                 t_max: float = 14.0,
                 dt: float = 1.0,
                 observation_noise: float = 1.0,
                 detection_limit: float = -0.65,
                 use_observations_only: bool = True,
                 seed: Optional[int] = None):
        """Initialize fast generator."""
        
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
        
        # Statistics
        self.total_simulations = 0
        self.failed_simulations = 0
        
    def generate_batch_fast(self, 
                           n_samples: int,
                           batch_size: int = 50,
                           show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch with optimized sequential processing.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        batch_size : int
            Size of processing chunks
        show_progress : bool
            Whether to show progress
            
        Returns:
        --------
        theta_batch : torch.Tensor
            Valid parameter vectors
        x_batch : torch.Tensor
            Valid observations
        """
        if show_progress:
            print(f"Generating {n_samples} TEIRV samples (fast sequential)...")
            
        start_time = time.time()
        
        # Collect results
        valid_theta_list = []
        valid_x_list = []
        
        # Process in chunks for memory efficiency and progress tracking
        n_chunks = (n_samples + batch_size - 1) // batch_size
        
        iterator = range(n_chunks)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing chunks")
            
        for chunk_idx in iterator:
            chunk_start = chunk_idx * batch_size
            chunk_end = min(chunk_start + batch_size, n_samples)
            chunk_size_actual = chunk_end - chunk_start
            
            # Sample parameters for this chunk
            theta_chunk = self.prior.sample((chunk_size_actual,)).numpy()
            
            # Fast batch simulation
            trajectories, success_mask = simulate_teirv_batch_fast(
                theta_batch=theta_chunk,
                initial_conditions=self.base_ic,
                t_max=self.t_max,
                t_grid=self.t_grid
            )
            
            # Filter successful simulations
            valid_theta_chunk = theta_chunk[success_mask]
            valid_trajectories = trajectories[success_mask]
            
            if len(valid_theta_chunk) > 0:
                # Apply observation model
                if self.use_observations_only:
                    # Extract viral loads and apply RT-PCR model
                    V_trajectories = valid_trajectories[:, :, 4]
                    observations = apply_observation_model_batch(
                        V_trajectories,
                        self.observation_noise,
                        self.detection_limit
                    )
                    valid_x_chunk = observations
                else:
                    # Use full trajectories
                    valid_x_chunk = valid_trajectories.reshape(len(valid_theta_chunk), -1)
                
                valid_theta_list.append(valid_theta_chunk)
                valid_x_list.append(valid_x_chunk)
            
            # Update statistics
            self.total_simulations += chunk_size_actual
            self.failed_simulations += chunk_size_actual - np.sum(success_mask)
        
        # Combine all chunks
        if len(valid_theta_list) == 0:
            raise RuntimeError("No valid simulations generated")
        
        all_theta = np.vstack(valid_theta_list)
        all_x = np.vstack(valid_x_list)
        
        # Convert to tensors
        theta_tensor = torch.tensor(all_theta, dtype=torch.float32)
        x_tensor = torch.tensor(all_x, dtype=torch.float32)
        
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
            'detection_limit': self.detection_limit
        }


def benchmark_fast_vs_original(n_samples: int = 50):
    """
    Benchmark fast implementation vs original.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples for comparison
    """
    print(f"Benchmarking Fast vs Original with {n_samples} samples")
    print("=" * 60)
    
    # Test original implementation
    print("\nOriginal implementation:")
    from teirv_data_generation import TEIRVDataGenerator
    
    original_gen = TEIRVDataGenerator(seed=42)
    start_time = time.time()
    
    try:
        theta_orig, x_orig = original_gen.generate_batch(
            n_samples=n_samples, 
            batch_size=10
        )
        orig_time = time.time() - start_time
        orig_speed = len(theta_orig) / orig_time
        print(f"✅ Original: {orig_time:.1f}s, {orig_speed:.2f} samples/sec")
        
    except Exception as e:
        print(f"❌ Original failed: {e}")
        orig_time = float('inf')
        orig_speed = 0
    
    # Test fast implementation
    print("\nFast implementation:")
    fast_gen = FastTEIRVDataGenerator(seed=42)
    start_time = time.time()
    
    try:
        theta_fast, x_fast = fast_gen.generate_batch_fast(
            n_samples=n_samples,
            batch_size=10
        )
        fast_time = time.time() - start_time
        fast_speed = len(theta_fast) / fast_time
        print(f"✅ Fast: {fast_time:.1f}s, {fast_speed:.2f} samples/sec")
        
    except Exception as e:
        print(f"❌ Fast failed: {e}")
        fast_time = float('inf')
        fast_speed = 0
    
    # Comparison
    if orig_time < float('inf') and fast_time < float('inf'):
        speedup = orig_time / fast_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster")
        print(f"Original time: {orig_time:.1f}s")
        print(f"Fast time: {fast_time:.1f}s")
    
    return {
        'original_time': orig_time,
        'fast_time': fast_time,
        'speedup': orig_time / fast_time if fast_time > 0 else 0
    }


if __name__ == '__main__':
    # Run benchmark
    benchmark_fast_vs_original(n_samples=20)