"""
Pure JAX implementation of TEIRV simulator for high-performance viral dynamics simulation.

Key features:
1. JIT-compiled Gillespie algorithm for fast stochastic simulation
2. GPU acceleration support with automatic device detection
3. Efficient grid-based output at specified time points
4. Vectorized batch processing for parameter inference
5. Pure JAX implementation for maximum performance

Main functions:
- gillespie_teirv_jax(): Single TEIRV simulation
- simulate_batch_jax(): Batch simulation for multiple parameter sets
- demo_teirv_jax(): Demonstration with GPU check and performance testing
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Tuple, Optional
from functools import partial


@jax.jit
def teirv_rates(state, theta):
    """
    Calculate TEIRV reaction rates.
    
    Parameters:
    -----------
    state : array of shape (5,)
        [T, E, I, R, V] populations
    theta : array of shape (6,)
        [β, π, δ, φ, ρ, V0] parameters
        
    Returns:
    --------
    rates : array of shape (7,)
        Reaction rates
    """
    T, E, I, R, V = state
    beta, pi, delta, phi, rho = theta[:5]
    k, c = 4.0, 10.0  # Fixed parameters
    
    rates = jnp.array([
        beta * 1e-9 * T * V,    # Infection: T + V → E + V
        phi * 1e-5 * T * I,     # Interferon: T + I → R + I
        rho * R,                # Reversion: R → T
        k * E,                  # Progression: E → I
        delta * I,              # Cell clearance: I → ∅
        pi * I,                 # Viral production: I → I + V
        c * V                   # Viral clearance: V → ∅
    ])
    
    return rates


@jax.jit
def teirv_stoichiometry():
    """
    Return TEIRV stoichiometry matrix.
    
    Returns:
    --------
    stoich : array of shape (7, 5)
        Stoichiometry matrix for [T, E, I, R, V] changes
    """
    return jnp.array([
        [-1,  1,  0,  0,  0],  # Infection: T→E
        [-1,  0,  0,  1,  0],  # Interferon: T→R
        [ 1,  0,  0, -1,  0],  # Reversion: R→T
        [ 0, -1,  1,  0,  0],  # Progression: E→I
        [ 0,  0, -1,  0,  0],  # Cell clearance: I→∅
        [ 0,  0,  0,  0,  1],  # Viral production: I→I+V
        [ 0,  0,  0,  0, -1]   # Viral clearance: V→∅
    ])


@jax.jit
def gillespie_step(state, t, theta, key):
    """
    Single Gillespie step.
    
    Parameters:
    -----------
    state : array of shape (5,)
        Current state [T, E, I, R, V]
    t : float
        Current time
    theta : array of shape (6,)
        Parameters
    key : JAX random key
        
    Returns:
    --------
    new_state : array of shape (5,)
        Updated state
    new_t : float
        Updated time
    stop : bool
        Whether to stop (rates = 0 or extinction)
    """
    rates = teirv_rates(state, theta)
    total_rate = jnp.sum(rates)
    
    # Check stopping conditions
    E, I, V = state[1], state[2], state[4]
    extinct = (E + I <= 1e-6) & (V <= 1e-6)
    no_reactions = total_rate <= 0
    stop = extinct | no_reactions
    
    # Sample time and reaction (even if stopping - keeps shapes consistent)
    key1, key2 = jr.split(key)
    tau = jr.exponential(key1) / jnp.maximum(total_rate, 1e-10)
    reaction_probs = rates / jnp.maximum(total_rate, 1e-10)
    reaction_idx = jr.choice(key2, 7, p=reaction_probs)
    
    # Apply reaction
    stoich = teirv_stoichiometry()
    state_change = stoich[reaction_idx]
    new_state = jnp.maximum(state + state_change, 0)  # Ensure non-negative
    new_t = t + tau
    
    # Return original state/time if stopping
    new_state = jnp.where(stop, state, new_state)
    new_t = jnp.where(stop, t, new_t)
    
    return new_state, new_t, stop


@partial(jax.jit, static_argnums=(4,))
def gillespie_simulate_to_grid(initial_state, theta, key, t_grid, max_iter=1000000):
    """
    Simulate TEIRV until time grid is covered.
    
    Parameters:
    -----------
    initial_state : array of shape (5,)
        Initial state [T, E, I, R, V]
    theta : array of shape (6,)
        Parameters [β, π, δ, φ, ρ, V0]
    key : JAX random key
    t_grid : array
        Target time points
    max_iter : int
        Maximum iterations (safety)
        
    Returns:
    --------
    output_states : array of shape (len(t_grid), 5)
        States at grid points
    """
    n_grid = len(t_grid)
    t_max = t_grid[-1]
    
    # Initialize output
    output_states = jnp.zeros((n_grid, 5))
    output_states = output_states.at[0].set(initial_state)
    
    # Simulation state
    current_state = initial_state
    t = 0.0
    grid_idx = 1
    iteration = 0
    
    def cond_fn(carry):
        _, _, grid_idx, t, iteration = carry
        return (t < t_max) & (grid_idx < n_grid) & (iteration < max_iter)
    
    def body_fn(carry):
        current_state, output_states, grid_idx, t, iteration = carry
        
        # Get next random key
        step_key = jr.fold_in(key, iteration)
        
        # Gillespie step
        new_state, new_t, stop = gillespie_step(current_state, t, theta, step_key)
        
        # Update grid points we've passed
        def update_grid(i, carry):
            grid_idx, output_states = carry
            # Check if we passed this grid point
            passed = (grid_idx < n_grid) & (t_grid[grid_idx] <= new_t)
            # Update if we passed it
            output_states = jnp.where(
                passed,
                output_states.at[grid_idx].set(current_state),
                output_states
            )
            grid_idx = jnp.where(passed, grid_idx + 1, grid_idx)
            return grid_idx, output_states
        
        # Update all grid points we might have passed
        grid_idx, output_states = jax.lax.fori_loop(
            0, n_grid - grid_idx, update_grid, (grid_idx, output_states)
        )
        
        # Stop if simulation ended
        t = jnp.where(stop, t_max, new_t)  # Force t >= t_max to exit
        current_state = jnp.where(stop, current_state, new_state)
        
        return current_state, output_states, grid_idx, t, iteration + 1
    
    initial_carry = (current_state, output_states, grid_idx, t, iteration)
    final_state, final_output, final_grid_idx, final_t, final_iter = jax.lax.while_loop(
        cond_fn, body_fn, initial_carry
    )
    
    # Fill remaining grid points with final state
    def fill_remaining(i, output_states):
        idx = final_grid_idx + i
        return jnp.where(
            idx < n_grid,
            output_states.at[idx].set(final_state),
            output_states
        )
    
    final_output = jax.lax.fori_loop(
        0, n_grid - final_grid_idx, fill_remaining, final_output
    )
    
    return final_output


def gillespie_teirv_jax(
    theta: np.ndarray,
    initial_conditions: dict,
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    key: Optional[jax.random.PRNGKey] = None,
    verbose: bool = False,
    return_jax_arrays: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JAX-based TEIRV simulation with automatic GPU detection.
    
    Parameters:
    -----------
    theta : array of shape (6,)
        Parameters [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Initial state
    t_max : float
        Maximum time
    t_grid : array, optional
        Target time points
    key : JAX random key, optional
    verbose : bool, optional
        Print device information
        
    Returns:
    --------
    times : array
        Time points
    states : array
        State trajectories
    """
    if key is None:
        key = jr.PRNGKey(0)
    
    # Print device info if requested
    if verbose:
        default_device = jax.devices()[0]
        print(f"Running simulation on: {default_device.device_kind} ({default_device.platform})")
    
    # Convert inputs to JAX arrays
    theta_jax = jnp.array(theta, dtype=jnp.float32)
    initial_state = jnp.array([
        initial_conditions['T'],
        initial_conditions['E'], 
        initial_conditions['I'],
        initial_conditions['R'],
        initial_conditions['V']
    ], dtype=jnp.float32)
    
    if t_grid is None:
        t_grid = jnp.linspace(0, t_max, 100)
    else:
        t_grid = jnp.array(t_grid, dtype=jnp.float32)
    
    # Run simulation
    output_states = gillespie_simulate_to_grid(
        initial_state, theta_jax, key, t_grid
    )
    
    if return_jax_arrays:
        return t_grid, output_states
    else:
        return np.array(t_grid), np.array(output_states)


@jax.jit
def simulate_batch_jax(theta_batch, initial_conditions_base, t_grid, key):
    """
    Vectorized batch simulation.
    
    Parameters:
    -----------
    theta_batch : array of shape (n_batch, 6)
        Batch of parameters
    initial_conditions_base : array of shape (5,)
        Base initial conditions (V will be overridden)
    t_grid : array
        Time grid
    key : JAX random key
        
    Returns:
    --------
    trajectories : array of shape (n_batch, len(t_grid), 5)
        Batch of trajectories
    """
    n_batch = theta_batch.shape[0]
    
    def simulate_single(i):
        theta_i = theta_batch[i]
        # Set V0 from parameters
        initial_state = initial_conditions_base.at[4].set(theta_i[5])
        key_i = jr.fold_in(key, i)
        return gillespie_simulate_to_grid(initial_state, theta_i, key_i, t_grid)
    
    return jax.vmap(simulate_single)(jnp.arange(n_batch))


def demo_teirv_jax():
    """Demonstrate JAX TEIRV simulator functionality with proper GPU timing."""
    import time
    import sys
    import os
    
    # Handle imports for both module and standalone usage
    try:
        from .teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid, check_gpu_availability
    except ImportError:
        # Add parent directory to path for standalone execution
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid, check_gpu_availability
    
    print("JAX TEIRV Simulator Demo")
    print("=" * 40)
    
    # Check GPU availability
    print("1. Checking GPU availability...")
    gpu_available = check_gpu_availability()
    print(f"   JAX backend: {jax.default_backend()}")
    print(f"   Available devices: {jax.devices()}")
    
    # Setup simulation
    print("\n2. Setting up simulation...")
    prior = create_teirv_prior(use_torch=False)
    ic = get_teirv_initial_conditions()
    t_grid = create_teirv_time_grid(14.0, 1.0, use_jax=False)
    key = jr.PRNGKey(42)
    
    # Sample parameters
    theta = np.array(prior.sample(key=key))
    ic['V'] = theta[5]
    
    print(f"   Parameters: β={theta[0]:.2f}, π={theta[1]:.0f}, δ={theta[2]:.2f}")
    print(f"               φ={theta[3]:.2f}, ρ={theta[4]:.2f}, V₀={theta[5]:.0f}")
    print(f"   Time grid: {len(t_grid)} points from 0 to {t_grid[-1]} days")
    
    # JIT warm-up to ensure compilation happens outside timing
    print("\n2.5. JIT warm-up...")
    warmup_grid = jnp.linspace(0, 1.0, 5)
    warmup_key = jr.PRNGKey(999)
    print("   Running small warm-up simulation...")
    _, warmup_states = gillespie_teirv_jax(theta, ic, 1.0, warmup_grid, warmup_key, return_jax_arrays=True)
    warmup_states.block_until_ready()  # Ensure warm-up completes
    print("   JIT compilation complete")
    
    # Run single simulation with proper GPU timing
    print("\n3. Running single simulation...")
    start = time.time()
    times_jax, states_jax = gillespie_teirv_jax(theta, ic, 14.0, t_grid, key, return_jax_arrays=True)
    # Synchronize GPU computation before measuring time
    states_jax.block_until_ready()
    sim_time = time.time() - start
    
    # Convert to numpy for display (outside timing)
    times = np.array(times_jax)
    states = np.array(states_jax)
    
    print(f"   Simulation completed in {sim_time:.3f}s")
    print(f"   Final state: T={states[-1,0]:.1e}, E={states[-1,1]:.0f}, I={states[-1,2]:.0f}")
    print(f"                R={states[-1,3]:.0f}, V={states[-1,4]:.1e}")
    
    # Test batch simulation with proper timing
    print("\n4. Testing batch simulation...")
    n_batch = 5
    theta_batch = jnp.array([prior.sample(key=jr.fold_in(key, i)) for i in range(n_batch)])
    ic_base = jnp.array([ic['T'], ic['E'], ic['I'], ic['R'], 0.0])
    
    # Warm-up batch simulation
    print("   Warming up batch simulation...")
    warmup_batch = simulate_batch_jax(theta_batch[:2], ic_base, warmup_grid, warmup_key)
    warmup_batch.block_until_ready()
    
    start = time.time()
    batch_results_jax = simulate_batch_jax(theta_batch, ic_base, jnp.array(t_grid), key)
    # Synchronize GPU computation before measuring time
    batch_results_jax.block_until_ready()
    batch_time = time.time() - start
    
    # Convert to numpy for display (outside timing)
    batch_results = np.array(batch_results_jax)
    
    print(f"   Batch ({n_batch} simulations) completed in {batch_time:.3f}s")
    print(f"   Per simulation: {batch_time/n_batch:.3f}s")
    print(f"   Batch output shape: {batch_results.shape}")
    
    # Performance summary
    print(f"\n5. Performance summary:")
    print(f"   Single simulation: {sim_time:.3f}s")
    print(f"   Batch per simulation: {batch_time/n_batch:.3f}s")
    batch_speedup = sim_time / (batch_time/n_batch)
    print(f"   Batch speedup: {batch_speedup:.1f}x")
    
    if gpu_available:
        print("   ✅ GPU acceleration enabled")
    else:
        print("   💻 Running on CPU")
    
    return times, states, batch_results


if __name__ == '__main__':
    demo_teirv_jax()
