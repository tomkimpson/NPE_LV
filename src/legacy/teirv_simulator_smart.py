"""
Smart TEIRV simulator with adaptive computational budgets.

Instead of arbitrary max_steps, we use:
1. Time-based budgets (max reactions per simulated day)
2. Rate-based early stopping (when rates become excessive)
3. Progress monitoring (detect stuck simulations)
"""
import numpy as np
import time
from typing import Tuple, Optional, Dict
import warnings


def gillespie_teirv_smart(
    theta: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    extinction_threshold: float = 1e-6,
    max_reactions_per_day: int = 50000,  # Budget: reactions per day
    max_rate_threshold: float = 1e8,     # Stop if rates become extreme
    timeout_seconds: float = 10.0        # Wall-clock timeout
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smart TEIRV simulation with adaptive computational budgets.
    
    Parameters:
    -----------
    theta : np.ndarray of shape (6,)
        Parameters [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Initial state
    t_max : float
        Maximum simulation time (days)
    t_grid : np.ndarray, optional
        Target time points
    extinction_threshold : float
        Population threshold for extinction
    max_reactions_per_day : int
        Computational budget (reactions per simulated day)
    max_rate_threshold : float
        Stop if total reaction rate exceeds this
    timeout_seconds : float
        Wall-clock time limit
        
    Returns:
    --------
    times : np.ndarray
        Time points
    states : np.ndarray
        State trajectories [T, E, I, R, V]
    """
    
    # Parse parameters
    beta, pi, delta, phi, rho = theta[:5]
    k, c = 4.0, 10.0
    
    # Initial state
    T = float(initial_conditions['T'])
    E = float(initial_conditions['E'])
    I = float(initial_conditions['I'])
    R = float(initial_conditions['R'])
    V = float(initial_conditions['V'])
    
    # Computational budget
    max_total_reactions = int(max_reactions_per_day * t_max)
    start_time = time.time()
    
    if t_grid is not None:
        return _simulate_smart_interpolated(
            T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
            t_max, t_grid, extinction_threshold,
            max_total_reactions, max_rate_threshold, timeout_seconds, start_time
        )
    else:
        return _simulate_smart_simple(
            T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
            t_max, extinction_threshold,
            max_total_reactions, max_rate_threshold, timeout_seconds, start_time
        )


def _simulate_smart_interpolated(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, t_grid, extinction_threshold,
    max_reactions, max_rate_threshold, timeout_seconds, start_time
) -> Tuple[np.ndarray, np.ndarray]:
    """Smart simulation with interpolated output."""
    
    n_grid = len(t_grid)
    output_states = np.zeros((n_grid, 5))
    
    # Initial state
    current_state = np.array([T, E, I, R, V], dtype=float)
    output_states[0] = current_state
    
    t = 0.0
    grid_idx = 1
    reaction_count = 0
    
    while t < t_max and grid_idx < n_grid and reaction_count < max_reactions:
        
        # Check wall-clock timeout
        if time.time() - start_time > timeout_seconds:
            warnings.warn(f"Simulation timeout after {timeout_seconds}s")
            break
            
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate reaction rates
        rates = _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c)
        total_rate = sum(rates)
        
        # Check for runaway rates
        if total_rate > max_rate_threshold:
            warnings.warn(f"Stopped simulation: rate {total_rate:.2e} exceeds threshold")
            break
            
        # Natural stopping condition
        if total_rate <= 0:
            break
            
        # Sample next reaction
        tau = np.random.exponential(1.0 / total_rate)
        t_next = t + tau
        
        # Record states at grid points we pass
        while grid_idx < n_grid and t_grid[grid_idx] <= t_next:
            output_states[grid_idx] = current_state.copy()
            grid_idx += 1
            
        if grid_idx >= n_grid:
            break
            
        # Execute reaction
        t = t_next
        T, E, I, R, V = _execute_reaction(T, E, I, R, V, rates, total_rate)
        current_state[:] = [T, E, I, R, V]
        
        reaction_count += 1
    
    # Fill remaining grid points
    while grid_idx < n_grid:
        output_states[grid_idx] = current_state
        grid_idx += 1
    
    return t_grid.copy(), output_states


def _simulate_smart_simple(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, extinction_threshold,
    max_reactions, max_rate_threshold, timeout_seconds, start_time
) -> Tuple[np.ndarray, np.ndarray]:
    """Smart simulation with simple output."""
    
    # Adaptive storage
    times = [0.0]
    states = [np.array([T, E, I, R, V], dtype=float)]
    
    t = 0.0
    reaction_count = 0
    sample_interval = max(1, max_reactions // 1000)  # Store ~1000 points max
    
    while t < t_max and reaction_count < max_reactions:
        
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            warnings.warn(f"Simulation timeout after {timeout_seconds}s")
            break
            
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate rates
        rates = _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c)
        total_rate = sum(rates)
        
        # Check runaway rates
        if total_rate > max_rate_threshold:
            warnings.warn(f"Stopped simulation: rate {total_rate:.2e} exceeds threshold")
            break
            
        if total_rate <= 0:
            break
            
        # Execute reaction
        tau = np.random.exponential(1.0 / total_rate)
        t += tau
        T, E, I, R, V = _execute_reaction(T, E, I, R, V, rates, total_rate)
        
        reaction_count += 1
        
        # Store state periodically
        if reaction_count % sample_interval == 0:
            times.append(t)
            states.append(np.array([T, E, I, R, V], dtype=float))
    
    # Store final state
    if len(times) == 0 or times[-1] != t:
        times.append(t)
        states.append(np.array([T, E, I, R, V], dtype=float))
    
    return np.array(times), np.array(states)


def _calculate_rates(T, E, I, R, V, beta, pi, delta, phi, rho, k, c):
    """Calculate reaction rates."""
    return [
        beta * 1e-9 * T * V,    # Infection
        phi * 1e-5 * T * I,     # Interferon
        rho * R,                # Reversion
        k * E,                  # Progression
        delta * I,              # Cell clearance
        pi * I,                 # Viral production
        c * V                   # Viral clearance
    ]


def _execute_reaction(T, E, I, R, V, rates, total_rate):
    """Execute a random reaction."""
    r = np.random.uniform(0, total_rate)
    cumulative = 0
    
    for i, rate in enumerate(rates):
        cumulative += rate
        if r < cumulative:
            if i == 0 and T >= 1:      # Infection
                T -= 1; E += 1
            elif i == 1 and T >= 1:    # Interferon
                T -= 1; R += 1
            elif i == 2 and R >= 1:    # Reversion
                R -= 1; T += 1
            elif i == 3 and E >= 1:    # Progression
                E -= 1; I += 1
            elif i == 4 and I >= 1:    # Cell clearance
                I -= 1
            elif i == 5 and I >= 1:    # Viral production
                V += 1
            elif i == 6 and V >= 1:    # Viral clearance
                V -= 1
            break
    
    return max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)


def benchmark_smart_implementation():
    """Test smart implementation with different computational budgets."""
    import time
    from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid
    
    print("Benchmarking Smart TEIRV Implementation")
    print("=" * 50)
    
    prior = create_teirv_prior()
    ic = get_teirv_initial_conditions()
    t_grid = create_teirv_time_grid(14.0, 1.0)
    
    # Test different computational budgets
    budgets = [10000, 25000, 50000, 100000]  # reactions per day
    
    for budget in budgets:
        print(f"\nTesting budget: {budget} reactions/day...")
        
        times = []
        successes = 0
        
        for i in range(5):
            theta = prior.sample().numpy()
            ic['V'] = theta[5]
            
            start = time.time()
            try:
                _, states = gillespie_teirv_smart(
                    theta, ic, 14.0, t_grid,
                    max_reactions_per_day=budget,
                    timeout_seconds=5.0  # Quick timeout for testing
                )
                elapsed = time.time() - start
                times.append(elapsed)
                successes += 1
                print(f"  Test {i+1}: {elapsed:.3f}s ✅")
                
            except Exception as e:
                elapsed = time.time() - start
                print(f"  Test {i+1}: {elapsed:.3f}s ❌ {e}")
        
        if times:
            avg_time = np.mean(times)
            print(f"  Average: {avg_time:.3f}s, Success rate: {successes}/5")
        else:
            print(f"  All tests failed")
    
    return True


if __name__ == '__main__':
    benchmark_smart_implementation()