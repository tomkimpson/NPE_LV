"""
Utility functions for TEIRV model NPE implementation.
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
from torch.distributions import LogNormal, Independent, Uniform, TransformedDistribution, ExpTransform
from torch.distributions.constraints import Constraint, real
import matplotlib.pyplot as plt


class TEIRVPrior(torch.distributions.Distribution):
    """
    Custom prior distribution for TEIRV parameters matching original paper.
    
    Handles the mixed Uniform and log-Uniform distributions used in 
    Germano et al. (2024) JSF paper.
    
    Inherits from torch.distributions.Distribution for SBI compatibility.
    """
    
    def __init__(self):
        # Parameter bounds from original paper config file
        self.beta_bounds = (0.0, 20.0)        # β: infection rate
        self.pi_bounds = (200.0, 600.0)       # π: virion production
        self.delta_bounds = (1.0, 11.0)       # δ: cell clearance  
        self.phi_bounds = (0.0, 15.0)         # φ: interferon protection
        self.rho_bounds = (0.0, 1.0)          # ρ: reversion rate
        self.lnv0_bounds = (0.0, 5.0)         # ln(V₀): log initial virions
        
        # Create individual distributions
        self.beta_dist = Uniform(*self.beta_bounds)
        self.pi_dist = Uniform(*self.pi_bounds)
        self.delta_dist = Uniform(*self.delta_bounds)
        self.phi_dist = Uniform(*self.phi_bounds)
        self.rho_dist = Uniform(*self.rho_bounds)
        self.lnv0_dist = Uniform(*self.lnv0_bounds)
        
        # Initialize parent Distribution class
        super().__init__(event_shape=torch.Size([6]), validate_args=False)
    
    def sample(self, sample_shape=torch.Size()):
        """Sample from the prior distribution."""
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
            
        # Sample each parameter
        beta = self.beta_dist.sample(sample_shape)
        pi = self.pi_dist.sample(sample_shape)
        delta = self.delta_dist.sample(sample_shape)
        phi = self.phi_dist.sample(sample_shape)
        rho = self.rho_dist.sample(sample_shape)
        lnv0 = self.lnv0_dist.sample(sample_shape)
        
        # Transform ln(V₀) to V₀
        v0 = torch.exp(lnv0)
        
        # Stack parameters: [β, π, δ, φ, ρ, V₀]
        if len(sample_shape) == 0:
            return torch.stack([beta, pi, delta, phi, rho, v0])
        else:
            return torch.stack([beta, pi, delta, phi, rho, v0], dim=-1)
    
    def log_prob(self, value):
        """Compute log probability of parameter values."""
        # Extract parameters
        beta, pi, delta, phi, rho, v0 = value.unbind(-1)
        
        # Compute log probabilities for each parameter
        log_prob = torch.zeros_like(beta)
        
        # Uniform distributions
        log_prob += self.beta_dist.log_prob(beta)
        log_prob += self.pi_dist.log_prob(pi)
        log_prob += self.delta_dist.log_prob(delta)
        log_prob += self.phi_dist.log_prob(phi)
        log_prob += self.rho_dist.log_prob(rho)
        
        # Log-uniform for V₀: p(V₀) = p(ln(V₀)) / V₀
        lnv0 = torch.log(v0)
        log_prob += self.lnv0_dist.log_prob(lnv0) - lnv0  # Jacobian correction
        
        return log_prob
    
    @property
    def support(self):
        """Return parameter support constraint for SBI compatibility."""
        # SBI expects a constraint object, not a dictionary
        return real
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds as dictionary for reference."""
        return {
            'beta': self.beta_bounds,
            'pi': self.pi_bounds, 
            'delta': self.delta_bounds,
            'phi': self.phi_bounds,
            'rho': self.rho_bounds,
            'v0': (np.exp(self.lnv0_bounds[0]), np.exp(self.lnv0_bounds[1]))
        }


def create_teirv_prior() -> TEIRVPrior:
    """
    Create prior distribution for TEIRV parameters.
    
    Based on actual priors used in Germano et al. (2024) JSF paper.
    Source: external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/config/cli-refractory-tiv-jsf.toml
    Lines 62-66 and 52:
    
    prior.beta = { name = "uniform", args.loc = 0, args.scale = 20}
    prior.phi = { name = "uniform", args.loc = 0, args.scale = 15}
    prior.rho = { name = "uniform", args.loc = 0, args.scale = 1}
    prior.delta = { name = "uniform", args.loc = 1.0, args.scale = 10}
    prior.pi = { name = "uniform", args.loc = 200, args.scale = 400}
    prior.lnV0 = { name = "uniform", args.loc = 0, args.scale = 5 }
    
    Parameter interpretations:
    - β (infection rate): Uniform(0, 20)
    - π (virion production): Uniform(200, 600)  
    - δ (cell clearance): Uniform(1, 11)
    - φ (interferon protection): Uniform(0, 15)
    - ρ (reversion rate): Uniform(0, 1)
    - V₀ (initial virions): exp(Uniform(0, 5)) ≈ [1, 148]
    
    Fixed parameters (not inferred):
    - k (E→I progression): 4
    - c (viral clearance): 10
    
    Returns:
    --------
    prior : TEIRVPrior
        Prior distribution over [β, π, δ, φ, ρ, V₀]
    """
    return TEIRVPrior()


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