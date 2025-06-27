"""
TEIRV (Target-Eclipsed-Infectious-Refractory-Virion) model simulation using Gillespie algorithm.

Based on the viral dynamics model from Germano et al. (2024):
"Jump-Switch-Flow for Non-Markovian stochastic PDMP models"

Model compartments:
- T: Target cells (susceptible to infection)
- E: Eclipsed cells (newly infected, not yet producing virus)  
- I: Infectious cells (actively producing virions)
- R: Refractory cells (temporarily resistant due to interferon)
- V: Virions (free virus particles)
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def gillespie_teirv(
    theta: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    max_steps: int = 100000,
    extinction_threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate TEIRV model using Gillespie algorithm.
    
    The model has 7 reactions:
    1. Infection: T + V → E + V, rate = β * T * V
    2. Interferon protection: T + I → R + I, rate = φ * T * I  
    3. Reversion: R → T, rate = ρ * R
    4. Progression: E → I, rate = k * E
    5. Cell clearance: I → ∅, rate = δ * I
    6. Viral production: I → I + V, rate = π * I
    7. Viral clearance: V → ∅, rate = c * V
    
    Parameters:
    -----------
    theta : np.ndarray of shape (6,) or dict
        Parameters [β, φ, ρ, k, δ, π, c] or subset for inference
        If array: [β, π, δ, φ, ρ, V0] (6 inferred parameters)
        If dict: can contain all parameters including fixed ones
    initial_conditions : dict
        Initial state: {'T': T0, 'E': E0, 'I': I0, 'R': R0, 'V': V0}
    t_max : float
        Maximum simulation time (days)
    t_grid : np.ndarray, optional
        Time points for interpolation. If None, returns raw trajectory
    max_steps : int
        Maximum number of simulation steps
    extinction_threshold : float
        Threshold below which populations are considered extinct
        
    Returns:
    --------
    times : np.ndarray
        Time points (either raw or interpolated)
    states : np.ndarray of shape (len(times), 5)
        State trajectories [T, E, I, R, V]
    """
    
    # Handle parameter input format
    if isinstance(theta, dict):
        # Direct parameter dictionary
        beta = theta['beta']
        phi = theta['phi'] 
        rho = theta['rho']
        k = theta['k']
        delta = theta['delta']
        pi = theta['pi']
        c = theta['c']
    else:
        # Array of inferred parameters + fixed parameters
        # Array format: [β, π, δ, φ, ρ, V0]
        beta = theta[0]
        pi = theta[1] 
        delta = theta[2]
        phi = theta[3]
        rho = theta[4]
        # V0 handled in initial_conditions
        
        # Fixed parameters (from paper)
        k = 4.0
        c = 10.0
    
    # Initial state
    T = float(initial_conditions['T'])
    E = float(initial_conditions['E']) 
    I = float(initial_conditions['I'])
    R = float(initial_conditions['R'])
    V = float(initial_conditions['V'])
    
    # Storage for trajectory
    times = [0.0]
    states = [(T, E, I, R, V)]
    
    t = 0.0
    step = 0
    
    while t < t_max and step < max_steps:
        # Check for extinction (all infected compartments near zero)
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            

        
        # Calculate reaction propensities
        # Scale β and φ as in original paper (see tiv.py lines 54-55)
        a1 = beta * 1e-9 * T * V           # Infection: T + V → E + V
        a2 = phi * 1e-5 * T * I           # Interferon: T + I → R + I  
        a3 = rho * R                      # Reversion: R → T
        a4 = k * E                        # Progression: E → I
        a5 = delta * I                    # Cell clearance: I → ∅
        a6 = pi * I                       # Viral production: I → I + V
        a7 = c * V                        # Viral clearance: V → ∅
        
        a_total = a1 + a2 + a3 + a4 + a5 + a6 + a7
        
        # If no reactions possible, stop
        if a_total <= 0:
            break
            
        # Sample time to next reaction
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        # Sample which reaction occurs
        r = np.random.uniform(0, a_total)
        
        if r < a1:
            # Infection: T + V → E + V (T decreases, E increases)
            if T >= 1:
                T -= 1
                E += 1
        elif r < a1 + a2:
            # Interferon protection: T + I → R + I (T decreases, R increases)
            if T >= 1:
                T -= 1
                R += 1
        elif r < a1 + a2 + a3:
            # Reversion: R → T
            if R >= 1:
                R -= 1
                T += 1
        elif r < a1 + a2 + a3 + a4:
            # Progression: E → I
            if E >= 1:
                E -= 1
                I += 1
        elif r < a1 + a2 + a3 + a4 + a5:
            # Cell clearance: I → ∅
            if I >= 1:
                I -= 1
        elif r < a1 + a2 + a3 + a4 + a5 + a6:
            # Viral production: I → I + V
            if I >= 1:
                V += 1
        else:
            # Viral clearance: V → ∅
            if V >= 1:
                V -= 1
        
        # Ensure non-negative populations
        T = max(0, T)
        E = max(0, E)
        I = max(0, I)
        R = max(0, R)
        V = max(0, V)
            
        times.append(t)
        states.append((T, E, I, R, V))
        step += 1
    

    print("Finished simulation")
    print(t,tau,t_max)

    # Convert to arrays
    times = np.array(times)
    states = np.array(states)
    
    # Interpolate to regular grid if requested
    if t_grid is not None:
        states_interp = interpolate_teirv_trajectory(times, states, t_grid)
        return t_grid, states_interp
    
    return times, states


def interpolate_teirv_trajectory(
    times: np.ndarray, 
    states: np.ndarray, 
    t_grid: np.ndarray
) -> np.ndarray:
    """
    Interpolate irregular TEIRV trajectory onto regular time grid.
    
    Parameters:
    -----------
    times : np.ndarray
        Original time points
    states : np.ndarray of shape (len(times), 5)
        State values [T, E, I, R, V] at original times
    t_grid : np.ndarray
        Target time grid for interpolation
        
    Returns:
    --------
    states_interp : np.ndarray of shape (len(t_grid), 5)
        Interpolated states
    """
    # Extend trajectory to cover full time range if needed
    if times[-1] < t_grid[-1]:
        # Pad with final values if simulation ended early
        times = np.append(times, t_grid[-1])
        states = np.vstack([states, states[-1:]])
    
    # Linear interpolation for each compartment
    states_interp = np.zeros((len(t_grid), 5))
    
    for i in range(5):  # T, E, I, R, V
        states_interp[:, i] = np.interp(t_grid, times, states[:, i])
    
    return states_interp


def simulate_teirv_batch(
    theta_batch: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Simulate multiple TEIRV trajectories in batch.
    
    Parameters:
    -----------
    theta_batch : np.ndarray of shape (n_batch, 6)
        Batch of parameter vectors [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Base initial conditions (V0 will be overridden by theta)
    t_max : float
        Maximum simulation time
    t_grid : np.ndarray
        Time grid for interpolation
    **kwargs : additional arguments for gillespie_teirv
        
    Returns:
    --------
    trajectories : np.ndarray of shape (n_batch, len(t_grid), 5)
        Batch of interpolated trajectories [T, E, I, R, V]
    """
    n_batch = theta_batch.shape[0]
    n_time = len(t_grid)
    trajectories = np.zeros((n_batch, n_time, 5))
    
    for i in range(n_batch):
        try:
            # Set V0 from theta
            ic = initial_conditions.copy()
            ic['V'] = theta_batch[i, 5]  # V0 is last parameter
            
            _, traj = gillespie_teirv(
                theta_batch[i], ic, t_max, t_grid, max_steps=100000, **kwargs
            )
            trajectories[i] = traj
            
        except Exception as e:
            warnings.warn(f"Simulation {i} failed: {e}. Using zeros.")
            trajectories[i] = np.zeros((n_time, 5))
            
    return trajectories


def check_teirv_trajectory_validity(trajectory: np.ndarray) -> bool:
    """
    Check if TEIRV trajectory is valid.
    
    Parameters:
    -----------
    trajectory : np.ndarray of shape (n_time, 5)
        State trajectory [T, E, I, R, V]
        
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
    if np.any(trajectory > 1e12):  # Higher threshold for viral loads
        return False
        
    return True