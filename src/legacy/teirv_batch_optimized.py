"""
Optimized batch processing for TEIRV simulations.

This module provides parallel and vectorized implementations 
to dramatically speed up batch data generation.
"""
import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Any
import multiprocessing as mp
from functools import partial
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .teirv_simulator import gillespie_teirv, check_teirv_trajectory_validity
from .teirv_utils import apply_observation_model


def simulate_single_teirv_worker(args: Tuple) -> Tuple[int, np.ndarray, bool]:
    """
    Worker function for parallel TEIRV simulation.
    
    Parameters:
    -----------
    args : tuple
        (index, theta, initial_conditions, t_max, t_grid, kwargs)
        
    Returns:
    --------
    index : int
        Simulation index
    trajectory : np.ndarray
        Simulated trajectory (zeros if failed)
    success : bool
        Whether simulation succeeded
    """
    index, theta, initial_conditions, t_max, t_grid, kwargs = args
    
    try:
        # Set V0 from parameters
        ic = initial_conditions.copy()
        ic['V'] = theta[5]  # V0 is last parameter
        
        # Simulate
        _, trajectory = gillespie_teirv(
            theta=theta,
            initial_conditions=ic, 
            t_max=t_max,
            t_grid=t_grid,
            **kwargs
        )
        
        # Check validity
        if check_teirv_trajectory_validity(trajectory):
            return index, trajectory, True
        else:
            return index, np.zeros((len(t_grid), 5)), False
            
    except Exception as e:
        warnings.warn(f"Simulation {index} failed: {e}")
        return index, np.zeros((len(t_grid), 5)), False


def simulate_teirv_batch_parallel(
    theta_batch: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: np.ndarray,
    n_processes: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate multiple TEIRV trajectories in parallel.
    
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
    n_processes : int, optional
        Number of parallel processes (default: CPU count)
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
    
    if n_processes is None:
        n_processes = min(n_batch, mp.cpu_count())
    
    # Prepare arguments for workers
    worker_args = [
        (i, theta_batch[i], initial_conditions, t_max, t_grid, kwargs)
        for i in range(n_batch)
    ]
    
    # Initialize results
    trajectories = np.zeros((n_batch, n_time, 5))
    success_mask = np.zeros(n_batch, dtype=bool)
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(simulate_single_teirv_worker, args): args[0] 
            for args in worker_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            try:
                index, trajectory, success = future.result()
                trajectories[index] = trajectory
                success_mask[index] = success
                
            except Exception as e:
                index = future_to_index[future]
                warnings.warn(f"Worker for simulation {index} failed: {e}")
                trajectories[index] = np.zeros((n_time, 5))
                success_mask[index] = False
    
    return trajectories, success_mask


def simulate_teirv_batch_chunked(
    theta_batch: np.ndarray,
    initial_conditions: Dict[str, float], 
    t_max: float,
    t_grid: np.ndarray,
    chunk_size: int = 10,
    n_processes: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate large batches using chunked parallel processing.
    
    Parameters:
    -----------
    theta_batch : np.ndarray of shape (n_batch, 6)
        Parameter vectors
    initial_conditions : dict
        Base initial conditions  
    t_max : float
        Maximum simulation time
    t_grid : np.ndarray
        Time grid
    chunk_size : int
        Size of chunks for parallel processing
    n_processes : int, optional
        Number of processes
    **kwargs : additional arguments
        
    Returns:
    --------
    trajectories : np.ndarray of shape (n_batch, len(t_grid), 5)
        All trajectories
    success_mask : np.ndarray of shape (n_batch,)
        Success indicators
    """
    n_batch = theta_batch.shape[0]
    n_time = len(t_grid)
    
    # Initialize results
    all_trajectories = np.zeros((n_batch, n_time, 5))
    all_success = np.zeros(n_batch, dtype=bool)
    
    # Process in chunks
    for i in range(0, n_batch, chunk_size):
        end_idx = min(i + chunk_size, n_batch)
        chunk_theta = theta_batch[i:end_idx]
        
        # Run chunk in parallel
        chunk_traj, chunk_success = simulate_teirv_batch_parallel(
            chunk_theta, initial_conditions, t_max, t_grid, 
            n_processes=n_processes, **kwargs
        )
        
        # Store results
        all_trajectories[i:end_idx] = chunk_traj
        all_success[i:end_idx] = chunk_success
    
    return all_trajectories, all_success


class OptimizedTEIRVDataGenerator:
    """
    Optimized data generator with parallel batch processing.
    
    Key improvements:
    1. Parallel simulation using multiprocessing
    2. Chunked processing for large batches
    3. Vectorized observation model application
    4. Memory-efficient batch operations
    """
    
    def __init__(self,
                 t_max: float = 14.0,
                 dt: float = 1.0,
                 observation_noise: float = 1.0,
                 detection_limit: float = -0.65,
                 use_observations_only: bool = True,
                 n_processes: Optional[int] = None,
                 chunk_size: int = 20,
                 seed: Optional[int] = None):
        """Initialize optimized generator."""
        
        # Import here to avoid circular imports
        from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid
        
        self.t_max = t_max
        self.dt = dt
        self.observation_noise = observation_noise
        self.detection_limit = detection_limit
        self.use_observations_only = use_observations_only
        self.n_processes = n_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.prior = create_teirv_prior()
        self.t_grid = create_teirv_time_grid(t_max, dt)
        self.base_ic = get_teirv_initial_conditions()
        
        # Statistics
        self.total_simulations = 0
        self.failed_simulations = 0
        
    def generate_batch_optimized(self, 
                                n_samples: int,
                                show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch with optimized parallel processing.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        show_progress : bool
            Whether to show progress updates
            
        Returns:
        --------
        theta_batch : torch.Tensor of shape (valid_samples, 6)
            Parameter vectors for valid simulations
        x_batch : torch.Tensor of shape (valid_samples, x_dim)
            Observations for valid simulations  
        """
        if show_progress:
            print(f"Generating {n_samples} TEIRV samples with {self.n_processes} processes...")
            
        start_time = time.time()
        
        # Sample parameters
        theta_batch = self.prior.sample((n_samples,)).numpy()
        
        # Run parallel simulations
        if show_progress:
            print("Running parallel simulations...")
            
        trajectories, success_mask = simulate_teirv_batch_chunked(
            theta_batch=theta_batch,
            initial_conditions=self.base_ic,
            t_max=self.t_max, 
            t_grid=self.t_grid,
            chunk_size=self.chunk_size,
            n_processes=self.n_processes
        )
        
        # Filter successful simulations
        valid_theta = theta_batch[success_mask]
        valid_trajectories = trajectories[success_mask]
        
        n_valid = len(valid_theta)
        n_failed = n_samples - n_valid
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"Completed {n_valid}/{n_samples} simulations in {elapsed:.1f}s")
            print(f"Success rate: {n_valid/n_samples:.1%}")
            print(f"Speed: {n_valid/elapsed:.1f} simulations/second")
        
        # Update statistics
        self.total_simulations += n_samples
        self.failed_simulations += n_failed
        
        if n_valid == 0:
            raise RuntimeError("No valid simulations generated")
        
        # Apply observation model vectorized
        if self.use_observations_only:
            if show_progress:
                print("Applying observation model...")
                
            # Extract viral load trajectories (V compartment)
            V_trajectories = valid_trajectories[:, :, 4]  # Shape: (n_valid, n_time)
            
            # Apply observation model to all trajectories at once
            observations = self._apply_observation_model_vectorized(V_trajectories)
            x_batch = observations
        else:
            # Use full trajectories (flatten last two dimensions)
            x_batch = valid_trajectories.reshape(n_valid, -1)
        
        # Convert to tensors
        theta_tensor = torch.tensor(valid_theta, dtype=torch.float32)
        x_tensor = torch.tensor(x_batch, dtype=torch.float32)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"Total generation time: {total_time:.1f}s")
            print(f"Output shapes: theta={theta_tensor.shape}, x={x_tensor.shape}")
        
        return theta_tensor, x_tensor
    
    def _apply_observation_model_vectorized(self, V_trajectories: np.ndarray) -> np.ndarray:
        """
        Apply RT-PCR observation model to multiple trajectories efficiently.
        
        Parameters:
        -----------
        V_trajectories : np.ndarray of shape (n_batch, n_time)
            Viral load trajectories
            
        Returns:
        --------
        observations : np.ndarray of shape (n_batch, n_time)
            RT-PCR observations
        """
        # Handle zero/negative viral loads
        V_positive = np.maximum(V_trajectories, 1e-10)
        
        # Log transform
        log_V = np.log10(V_positive)
        
        # Add observation noise (vectorized)
        noise = np.random.normal(0, self.observation_noise, size=log_V.shape)
        observations = log_V + noise
        
        # Apply detection limit
        observations = np.maximum(observations, self.detection_limit)
        
        return observations
    
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
            'n_processes': self.n_processes,
            'chunk_size': self.chunk_size
        }


def benchmark_batch_generation(n_samples: int = 50, n_processes_list: List[int] = None):
    """
    Benchmark different batch generation approaches.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples for benchmarking
    n_processes_list : list
        List of process counts to test
    """
    if n_processes_list is None:
        n_processes_list = [1, 2, 4, mp.cpu_count()]
    
    print(f"Benchmarking batch generation with {n_samples} samples...")
    print("=" * 60)
    
    results = {}
    
    for n_proc in n_processes_list:
        print(f"\nTesting with {n_proc} process(es):")
        
        generator = OptimizedTEIRVDataGenerator(
            n_processes=n_proc,
            chunk_size=10,
            seed=42
        )
        
        start_time = time.time()
        
        try:
            theta, x = generator.generate_batch_optimized(
                n_samples=n_samples,
                show_progress=True
            )
            
            elapsed = time.time() - start_time
            speed = len(theta) / elapsed
            
            results[n_proc] = {
                'time': elapsed,
                'speed': speed,
                'success_rate': len(theta) / n_samples,
                'valid_samples': len(theta)
            }
            
            print(f"✅ Completed: {elapsed:.1f}s, {speed:.1f} samples/sec")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[n_proc] = None
    
    # Summary
    print(f"\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Processes':<10} {'Time (s)':<10} {'Speed':<15} {'Success Rate':<12}")
    print("-" * 60)
    
    for n_proc, result in results.items():
        if result:
            print(f"{n_proc:<10} {result['time']:<10.1f} "
                  f"{result['speed']:<15.1f} {result['success_rate']:<12.1%}")
        else:
            print(f"{n_proc:<10} {'FAILED':<10}")
    
    return results


if __name__ == '__main__':
    # Run benchmark
    benchmark_batch_generation(n_samples=20)