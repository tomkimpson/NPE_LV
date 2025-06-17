"""
Utility functions for TEIRV model NPE implementation.
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
from torch.distributions import LogNormal, Independent
import matplotlib.pyplot as plt


def create_teirv_prior() -> Independent:
    """
    Create LogNormal prior distribution for TEIRV parameters.
    
    Based on Table in Germano et al. (2024):
    - β (infection rate): LogNormal(log(2.5×10⁻⁹), 1.0)
    - π (virion production): LogNormal(log(10²), 1.0)  
    - δ (cell clearance): LogNormal(log(0.5), 1.0)
    - φ (interferon protection): LogNormal(log(10⁻⁹), 2.0)
    - ρ (reversion rate): LogNormal(log(0.1), 1.0)
    - V₀ (initial virions): LogNormal(log(10³), 1.0)
    
    Fixed parameters (not inferred):
    - k (E→I progression): 4
    - c (viral clearance): 10
    
    Returns:
    --------
    prior : torch.distributions.Independent
        Prior over [β, π, δ, φ, ρ, V₀]
    """
    # Define individual LogNormal distributions
    log_means = torch.tensor([
        np.log(2.5e-9),  # β
        np.log(1e2),     # π
        np.log(0.5),     # δ  
        np.log(1e-9),    # φ
        np.log(0.1),     # ρ
        np.log(1e3)      # V₀
    ])
    
    log_stds = torch.tensor([
        1.0,  # β
        1.0,  # π
        1.0,  # δ
        2.0,  # φ (wider uncertainty)
        1.0,  # ρ
        1.0   # V₀
    ])
    
    # Create multivariate LogNormal
    lognormal_dist = LogNormal(log_means, log_stds)
    return Independent(lognormal_dist, 1)


def get_teirv_initial_conditions(V0: float = 1e3) -> Dict[str, float]:
    """
    Get standard initial conditions for TEIRV model.
    
    From Germano et al. (2024):
    - T(0) = 8×10⁷ (target cells)
    - E(0) = 1 (eclipsed cells)
    - I(0) = 0 (infectious cells)
    - R(0) = 0 (refractory cells)
    - V(0) = inferred parameter
    
    Parameters:
    -----------
    V0 : float
        Initial virion count (will be overridden during inference)
        
    Returns:
    --------
    initial_conditions : dict
        Dictionary of initial conditions
    """
    return {
        'T': 8e7,
        'E': 1.0,
        'I': 0.0,
        'R': 0.0,
        'V': V0
    }


def get_teirv_fixed_parameters() -> Dict[str, float]:
    """
    Get fixed parameters for TEIRV model.
    
    Returns:
    --------
    fixed_params : dict
        Dictionary of fixed parameters
    """
    return {
        'k': 4.0,   # E → I progression rate
        'c': 10.0   # Viral clearance rate
    }


def create_teirv_time_grid(t_max: float = 14.0, dt: float = 1.0) -> np.ndarray:
    """
    Create time grid for TEIRV simulation matching clinical data.
    
    Clinical data has 14 daily measurements.
    
    Parameters:
    -----------
    t_max : float
        Maximum time (days)
    dt : float
        Time step (days)
        
    Returns:
    --------
    t_grid : np.ndarray
        Time grid
    """
    return np.arange(0, t_max + dt, dt)


def apply_observation_model(
    V_trajectory: np.ndarray,
    sigma: float = 1.0,
    detection_limit: float = -0.65,
    add_noise: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply RT-PCR observation model to viral load trajectory.
    
    Based on Germano et al. (2024):
    1. Transform: log₁₀(V) 
    2. Add Gaussian noise: y ~ N(log₁₀(V), σ²)
    3. Apply detection limit: y ≥ detection_limit
    
    Parameters:
    -----------
    V_trajectory : np.ndarray
        Virion counts over time
    sigma : float
        Observation noise standard deviation
    detection_limit : float
        Lower detection limit for RT-PCR
    add_noise : bool
        Whether to add observation noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    observations : np.ndarray
        RT-PCR observations (log₁₀ scale)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Handle zero/negative viral loads
    V_positive = np.maximum(V_trajectory, 1e-10)
    
    # Log transform
    log_V = np.log10(V_positive)
    
    # Add observation noise
    if add_noise:
        noise = np.random.normal(0, sigma, size=log_V.shape)
        observations = log_V + noise
    else:
        observations = log_V.copy()
    
    # Apply detection limit
    observations = np.maximum(observations, detection_limit)
    
    return observations


def cn_to_log_viral_load(cn_values: np.ndarray) -> np.ndarray:
    """
    Convert RT-PCR Cycle Number (CN) to log₁₀ viral load.
    
    Based on calibration: log₁₀(V) = 11.35 - 0.25 × CN
    
    Parameters:
    -----------
    cn_values : np.ndarray
        RT-PCR cycle numbers
        
    Returns:
    --------
    log_viral_load : np.ndarray
        Log₁₀ viral load values
    """
    return 11.35 - 0.25 * cn_values


def preprocess_clinical_data(
    observations: np.ndarray,
    detection_limit: float = -0.65
) -> np.ndarray:
    """
    Preprocess clinical RT-PCR data for NPE.
    
    Parameters:
    -----------
    observations : np.ndarray
        Raw log₁₀ viral load observations
    detection_limit : float
        Detection limit value
        
    Returns:
    --------
    processed_obs : np.ndarray
        Processed observations
    """
    # Apply detection limit
    processed = np.maximum(observations, detection_limit)
    
    # Could add normalization here if needed for neural network
    # For now, keep in original scale
    
    return processed


def visualize_teirv_trajectory(
    times: np.ndarray,
    states: np.ndarray,
    observations: Optional[np.ndarray] = None,
    true_params: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize TEIRV model trajectory and observations.
    
    Parameters:
    -----------
    times : np.ndarray
        Time points
    states : np.ndarray of shape (len(times), 5)
        State trajectories [T, E, I, R, V]
    observations : np.ndarray, optional
        RT-PCR observations
    true_params : np.ndarray, optional
        True parameter values for title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    compartment_names = ['Target (T)', 'Eclipsed (E)', 'Infectious (I)', 
                        'Refractory (R)', 'Virions (V)']
    colors = ['blue', 'orange', 'red', 'purple', 'green']
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each compartment
    for i, (name, color) in enumerate(zip(compartment_names, colors)):
        ax = axes[i]
        ax.plot(times, states[:, i], color=color, linewidth=2, label=name)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Population')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Log scale for large compartments
        if i in [0, 4]:  # T and V can be very large
            ax.set_yscale('log')
    
    # Plot viral load observations if provided
    if observations is not None:
        ax = axes[5]
        ax.plot(times, np.log10(np.maximum(states[:, 4], 1e-10)), 
               'g-', linewidth=2, label='True log₁₀(V)')
        ax.scatter(times, observations, color='red', s=50, 
                  label='RT-PCR observations', zorder=5)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('log₁₀ Viral Load')
        ax.set_title('Viral Load & Observations')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add parameter info to title if provided
    if true_params is not None:
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        param_str = ', '.join([f'{name}={val:.2e}' for name, val 
                              in zip(param_names, true_params)])
        fig.suptitle(f'TEIRV Model Trajectory\n{param_str}', fontsize=12)
    else:
        fig.suptitle('TEIRV Model Trajectory', fontsize=14)
    
    plt.tight_layout()
    return fig


def teirv_parameter_summary(samples: torch.Tensor) -> Dict[str, Any]:
    """
    Compute summary statistics for TEIRV parameter samples.
    
    Parameters:
    -----------
    samples : torch.Tensor of shape (n_samples, 6)
        Parameter samples [β, π, δ, φ, ρ, V₀]
        
    Returns:
    --------
    summary : dict
        Dictionary with mean, std, quantiles for each parameter
    """
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    samples_np = samples.numpy()
    
    summary = {}
    for i, name in enumerate(param_names):
        param_samples = samples_np[:, i]
        summary[name] = {
            'mean': np.mean(param_samples),
            'std': np.std(param_samples),
            'median': np.median(param_samples),
            'q025': np.percentile(param_samples, 2.5),
            'q975': np.percentile(param_samples, 97.5)
        }
    
    return summary