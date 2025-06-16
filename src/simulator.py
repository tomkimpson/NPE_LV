"""
Gillespie algorithm implementation for stochastic Lotka-Volterra model.
"""
import numpy as np
from typing import Tuple, Optional
import warnings


def gillespie_lv(
    theta: np.ndarray,
    x0: Tuple[int, int],
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    max_steps: int = 100000,
    extinction_threshold: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate stochastic Lotka-Volterra model using Gillespie algorithm.
    
    The model has four reactions:
    1. Prey birth: x1 -> x1 + 1, rate = alpha * x1
    2. Predation: x1 -> x1 - 1, x2 -> x2 + 1, rate = beta * x1 * x2  
    3. Predator birth: x2 -> x2 + 1, rate = delta * x1 * x2
    4. Predator death: x2 -> x2 - 1, rate = gamma * x2
    
    Parameters:
    -----------
    theta : np.ndarray of shape (4,)
        Parameters [alpha, beta, delta, gamma]
    x0 : tuple of int
        Initial conditions (prey, predator)
    t_max : float
        Maximum simulation time
    t_grid : np.ndarray, optional
        Time points for interpolation. If None, returns raw trajectory
    max_steps : int
        Maximum number of simulation steps
    extinction_threshold : int
        Population threshold below which species is considered extinct
        
    Returns:
    --------
    times : np.ndarray
        Time points (either raw or interpolated)
    populations : np.ndarray of shape (len(times), 2)
        Population trajectories [prey, predator]
    """
    alpha, beta, delta, gamma = theta
    x1, x2 = x0
    
    # Storage for trajectory
    times = [0.0]
    populations = [(x1, x2)]
    
    t = 0.0
    step = 0
    
    while t < t_max and step < max_steps:
        # Check for extinction
        if x1 <= extinction_threshold or x2 <= extinction_threshold:
            break
            
        # Calculate reaction propensities
        a1 = alpha * x1                    # prey birth
        a2 = beta * x1 * x2               # predation
        a3 = delta * x1 * x2              # predator birth  
        a4 = gamma * x2                   # predator death
        
        a_total = a1 + a2 + a3 + a4
        
        # If no reactions possible, stop
        if a_total <= 0:
            break
            
        # Sample time to next reaction
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        
        # Sample which reaction occurs
        r = np.random.uniform(0, a_total)
        
        if r < a1:
            # Prey birth
            x1 += 1
        elif r < a1 + a2:
            # Predation
            x1 -= 1
            x2 += 1
        elif r < a1 + a2 + a3:
            # Predator birth
            x2 += 1
        else:
            # Predator death
            x2 -= 1
            
        times.append(t)
        populations.append((x1, x2))
        step += 1
    
    # Convert to arrays
    times = np.array(times)
    populations = np.array(populations)
    
    # Interpolate to regular grid if requested
    if t_grid is not None:
        populations_interp = interpolate_trajectory(times, populations, t_grid)
        return t_grid, populations_interp
    
    return times, populations


def interpolate_trajectory(
    times: np.ndarray, 
    populations: np.ndarray, 
    t_grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate irregular trajectory onto regular time grid.
    
    Parameters:
    -----------
    times : np.ndarray
        Original time points
    populations : np.ndarray of shape (len(times), 2)
        Population values at original times
    t_grid : np.ndarray
        Target time grid for interpolation
        
    Returns:
    --------
    populations_interp : np.ndarray of shape (len(t_grid), 2)
        Interpolated populations
    """
    # Extend trajectory to cover full time range if needed
    if times[-1] < t_grid[-1]:
        # Pad with final values if simulation ended early
        times = np.append(times, t_grid[-1])
        populations = np.vstack([populations, populations[-1:]])
    
    # Linear interpolation for each species
    populations_interp = np.zeros((len(t_grid), 2))
    
    for i in range(2):  # prey and predator
        populations_interp[:, i] = np.interp(t_grid, times, populations[:, i])
    
    return populations_interp


def simulate_lv_batch(
    theta_batch: np.ndarray,
    x0: Tuple[int, int],
    t_max: float,
    t_grid: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Simulate multiple LV trajectories in batch.
    
    Parameters:
    -----------
    theta_batch : np.ndarray of shape (n_batch, 4)
        Batch of parameter vectors
    x0 : tuple of int
        Initial conditions
    t_max : float
        Maximum simulation time
    t_grid : np.ndarray
        Time grid for interpolation
    **kwargs : additional arguments for gillespie_lv
        
    Returns:
    --------
    trajectories : np.ndarray of shape (n_batch, len(t_grid), 2)
        Batch of interpolated trajectories
    """
    n_batch = theta_batch.shape[0]
    n_time = len(t_grid)
    trajectories = np.zeros((n_batch, n_time, 2))
    
    for i in range(n_batch):
        try:
            _, traj = gillespie_lv(theta_batch[i], x0, t_max, t_grid, **kwargs)
            trajectories[i] = traj
        except Exception as e:
            warnings.warn(f"Simulation {i} failed: {e}. Using zeros.")
            trajectories[i] = np.zeros((n_time, 2))
            
    return trajectories