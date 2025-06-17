"""
Clean TEIRV simulator following standard Gillespie implementation patterns.

Key improvements over original:
1. No artificial max_steps limit
2. Optimized data storage (sparse grid sampling)
3. Clean, readable code structure
4. Only store what we need
"""
import numpy as np
from typing import Tuple, Optional, Dict
import warnings


def gillespie_teirv_clean(
    theta: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    extinction_threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean TEIRV simulation using standard Gillespie algorithm.
    
    Parameters:
    -----------
    theta : np.ndarray of shape (6,)
        Parameters [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Initial state: {'T': T0, 'E': E0, 'I': I0, 'R': R0, 'V': V0}
    t_max : float
        Maximum simulation time (days)
    t_grid : np.ndarray, optional
        Target time points for interpolation
    extinction_threshold : float
        Population threshold for extinction
        
    Returns:
    --------
    times : np.ndarray
        Time points
    states : np.ndarray of shape (len(times), 5)
        State trajectories [T, E, I, R, V]
    """
    
    # Parse parameters
    beta, pi, delta, phi, rho = theta[:5]
    k, c = 4.0, 10.0  # Fixed parameters
    
    # Initial state
    T = float(initial_conditions['T'])
    E = float(initial_conditions['E'])
    I = float(initial_conditions['I'])
    R = float(initial_conditions['R'])
    V = float(initial_conditions['V'])
    
    if t_grid is not None:
        # Use sparse storage for interpolated output
        return _simulate_with_interpolation(
            T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
            t_max, t_grid, extinction_threshold
        )
    else:
        # Simple simulation without interpolation
        return _simulate_simple(
            T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
            t_max, extinction_threshold
        )


def _simulate_with_interpolation(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, t_grid, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate with sparse storage at target time points only.
    
    This avoids storing every single reaction event, dramatically
    improving performance for typical use cases.
    """
    n_grid = len(t_grid)
    output_states = np.zeros((n_grid, 5))
    
    # Set initial state
    current_state = np.array([T, E, I, R, V], dtype=float)
    output_states[0] = current_state
    
    t = 0.0
    grid_idx = 1  # Next grid point to fill
    
    # Standard Gillespie loop - no artificial step limit!
    while t < t_max and grid_idx < n_grid:
        # Check extinction (natural stopping condition)
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate reaction propensities
        rates = _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c)
        total_rate = sum(rates)
        
        # Natural stopping condition - no more reactions possible
        if total_rate <= 0:
            break
            
        # Sample time to next reaction
        tau = np.random.exponential(1.0 / total_rate)
        t_next = t + tau
        
        # Record states at any grid points we pass
        while grid_idx < n_grid and t_grid[grid_idx] <= t_next:
            output_states[grid_idx] = current_state.copy()
            grid_idx += 1
            
        if grid_idx >= n_grid:
            break
            
        # Advance time
        t = t_next
        
        # Sample and execute reaction
        T, E, I, R, V = _execute_reaction(T, E, I, R, V, rates, total_rate)
        current_state[:] = [T, E, I, R, V]
    
    # Fill remaining grid points with final state
    while grid_idx < n_grid:
        output_states[grid_idx] = current_state
        grid_idx += 1
    
    return t_grid.copy(), output_states


def _simulate_simple(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple simulation storing periodic snapshots (not every reaction).
    """
    # Store snapshots every ~1000 reactions for memory efficiency
    snapshot_interval = 1000
    max_snapshots = 10000
    
    times = np.zeros(max_snapshots)
    states = np.zeros((max_snapshots, 5))
    
    # Initial state
    current_state = np.array([T, E, I, R, V], dtype=float)
    times[0] = 0.0
    states[0] = current_state
    
    t = 0.0
    reaction_count = 0
    snapshot_idx = 1
    
    # Standard Gillespie loop
    while t < t_max:
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate rates
        rates = _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c)
        total_rate = sum(rates)
        
        if total_rate <= 0:
            break
            
        # Sample reaction
        tau = np.random.exponential(1.0 / total_rate)
        t += tau
        
        # Execute reaction
        T, E, I, R, V = _execute_reaction(T, E, I, R, V, rates, total_rate)
        current_state[:] = [T, E, I, R, V]
        
        reaction_count += 1
        
        # Store snapshot periodically
        if (reaction_count % snapshot_interval == 0 and 
            snapshot_idx < max_snapshots):
            times[snapshot_idx] = t
            states[snapshot_idx] = current_state.copy()
            snapshot_idx += 1
    
    # Store final state
    if snapshot_idx < max_snapshots:
        times[snapshot_idx] = t
        states[snapshot_idx] = current_state.copy()
        snapshot_idx += 1
    
    return times[:snapshot_idx], states[:snapshot_idx]


def _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c):
    """Calculate all reaction rates."""
    return [
        beta * 1e-9 * T * V,    # Infection: T + V → E + V
        phi * 1e-5 * T * I,     # Interferon: T + I → R + I  
        rho * R,                # Reversion: R → T
        k * E,                  # Progression: E → I
        delta * I,              # Cell clearance: I → ∅
        pi * I,                 # Viral production: I → I + V
        c * V                   # Viral clearance: V → ∅
    ]


def _execute_reaction(T, E, I, R, V, rates, total_rate):
    """Sample and execute a reaction."""
    # Sample which reaction occurs
    r = np.random.uniform(0, total_rate)
    cumulative = 0
    
    for i, rate in enumerate(rates):
        cumulative += rate
        if r < cumulative:
            # Execute reaction i
            if i == 0:  # Infection: T + V → E + V
                if T >= 1:
                    T -= 1
                    E += 1
            elif i == 1:  # Interferon: T + I → R + I
                if T >= 1:
                    T -= 1
                    R += 1
            elif i == 2:  # Reversion: R → T
                if R >= 1:
                    R -= 1
                    T += 1
            elif i == 3:  # Progression: E → I
                if E >= 1:
                    E -= 1
                    I += 1
            elif i == 4:  # Cell clearance: I → ∅
                if I >= 1:
                    I -= 1
            elif i == 5:  # Viral production: I → I + V
                if I >= 1:
                    V += 1
            elif i == 6:  # Viral clearance: V → ∅
                if V >= 1:
                    V -= 1
            break
    
    # Ensure non-negative populations
    return max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)


def simulate_teirv_batch_clean(
    theta_batch: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Clean batch simulation without max_steps.
    """
    n_batch = theta_batch.shape[0]
    n_time = len(t_grid)
    trajectories = np.zeros((n_batch, n_time, 5))
    
    for i in range(n_batch):
        try:
            # Set V0 from parameters
            ic = initial_conditions.copy()
            ic['V'] = theta_batch[i, 5]
            
            _, traj = gillespie_teirv_clean(
                theta_batch[i], ic, t_max, t_grid, **kwargs
            )
            trajectories[i] = traj
            
        except Exception as e:
            warnings.warn(f"Simulation {i} failed: {e}. Using zeros.")
            trajectories[i] = np.zeros((n_time, 5))
            
    return trajectories


def benchmark_clean_vs_original():
    """Compare clean implementation with original."""
    import time
    from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid
    from teirv_simulator import gillespie_teirv
    
    print("Benchmarking Clean vs Original TEIRV Implementation")
    print("=" * 60)
    
    # Test setup
    prior = create_teirv_prior()
    ic = get_teirv_initial_conditions()
    t_grid = create_teirv_time_grid(14.0, 1.0)
    n_tests = 5
    
    # Test original
    print("Testing original implementation...")
    orig_times = []
    for i in range(n_tests):
        theta = prior.sample().numpy()
        ic['V'] = theta[5]
        
        start = time.time()
        try:
            _, states = gillespie_teirv(theta, ic, 14.0, t_grid)
            elapsed = time.time() - start
            orig_times.append(elapsed)
            print(f"  Test {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  Test {i+1}: FAILED - {e}")
    
    # Test clean
    print(f"\nTesting clean implementation...")
    clean_times = []
    for i in range(n_tests):
        theta = prior.sample().numpy()
        ic['V'] = theta[5]
        
        start = time.time()
        try:
            _, states = gillespie_teirv_clean(theta, ic, 14.0, t_grid)
            elapsed = time.time() - start
            clean_times.append(elapsed)
            print(f"  Test {i+1}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  Test {i+1}: FAILED - {e}")
    
    # Compare
    if orig_times and clean_times:
        orig_avg = np.mean(orig_times)
        clean_avg = np.mean(clean_times)
        speedup = orig_avg / clean_avg
        
        print(f"\nResults:")
        print(f"  Original: {orig_avg:.3f} ± {np.std(orig_times):.3f}s")
        print(f"  Clean:    {clean_avg:.3f} ± {np.std(clean_times):.3f}s")
        print(f"  Speedup:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
    
    return orig_times, clean_times


if __name__ == '__main__':
    benchmark_clean_vs_original()