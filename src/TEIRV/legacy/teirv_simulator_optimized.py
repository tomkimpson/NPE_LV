"""
Optimized TEIRV simulator with efficient data storage strategies.

Key optimizations:
1. Pre-allocated arrays instead of dynamic lists
2. Sparse storage - only save data at target time points
3. Streaming interpolation - interpolate on-the-fly
4. Memory-efficient trajectory storage
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def gillespie_teirv_optimized(
    theta: np.ndarray,
    initial_conditions: Dict[str, float],
    t_max: float,
    t_grid: Optional[np.ndarray] = None,
    max_steps: int = 1000000,  # Keep high since we're not storing everything
    extinction_threshold: float = 1e-6,
    storage_strategy: str = "sparse"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-optimized TEIRV simulation using Gillespie algorithm.
    
    Storage strategies:
    - "sparse": Only store states at target time points (fastest)
    - "streaming": Interpolate on-the-fly during simulation  
    - "adaptive": Pre-allocated arrays with intelligent resizing
    
    Parameters:
    -----------
    theta : np.ndarray of shape (6,)
        Parameters [β, π, δ, φ, ρ, V0]
    initial_conditions : dict
        Initial state: {'T': T0, 'E': E0, 'I': I0, 'R': R0, 'V': V0}
    t_max : float
        Maximum simulation time (days)
    t_grid : np.ndarray, optional
        Target time points for output
    max_steps : int
        Maximum reaction events (can be high with optimized storage)
    extinction_threshold : float
        Population threshold for extinction
    storage_strategy : str
        Storage optimization method
        
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
        if storage_strategy == "sparse":
            return _gillespie_sparse_storage(
                T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
                t_max, t_grid, max_steps, extinction_threshold
            )
        elif storage_strategy == "streaming":
            return _gillespie_streaming_interpolation(
                T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
                t_max, t_grid, max_steps, extinction_threshold
            )
        else:  # adaptive
            return _gillespie_adaptive_storage(
                T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
                t_max, t_grid, max_steps, extinction_threshold
            )
    else:
        # No interpolation needed - use minimal storage
        return _gillespie_minimal_storage(
            T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
            t_max, max_steps, extinction_threshold
        )


def _gillespie_sparse_storage(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, t_grid, max_steps, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse storage: Only save states at target time grid points.
    
    This is the fastest method for typical use cases where we only
    need interpolated output at specific time points.
    """
    n_grid = len(t_grid)
    output_states = np.zeros((n_grid, 5))
    
    # Set initial state
    current_state = np.array([T, E, I, R, V], dtype=float)
    output_states[0] = current_state
    
    t = 0.0
    step = 0
    grid_idx = 1  # Next grid point to fill
    
    while t < t_max and step < max_steps and grid_idx < n_grid:
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate reaction rates
        a1 = beta * 1e-9 * T * V
        a2 = phi * 1e-5 * T * I
        a3 = rho * R
        a4 = k * E
        a5 = delta * I
        a6 = pi * I
        a7 = c * V
        
        a_total = a1 + a2 + a3 + a4 + a5 + a6 + a7
        
        if a_total <= 0:
            break
            
        # Sample next reaction time
        tau = np.random.exponential(1.0 / a_total)
        t_next = t + tau
        
        # Check if we need to record state(s) before this reaction
        while grid_idx < n_grid and t_grid[grid_idx] <= t_next:
            # Record state at this grid point (using current state)
            output_states[grid_idx] = current_state
            grid_idx += 1
            
        if grid_idx >= n_grid:
            break
            
        # Advance time
        t = t_next
        
        # Execute reaction
        r = np.random.uniform(0, a_total)
        
        if r < a1:  # Infection
            if T >= 1: T -= 1; E += 1
        elif r < a1 + a2:  # Interferon protection
            if T >= 1: T -= 1; R += 1
        elif r < a1 + a2 + a3:  # Reversion
            if R >= 1: R -= 1; T += 1
        elif r < a1 + a2 + a3 + a4:  # Progression
            if E >= 1: E -= 1; I += 1
        elif r < a1 + a2 + a3 + a4 + a5:  # Cell clearance
            if I >= 1: I -= 1
        elif r < a1 + a2 + a3 + a4 + a5 + a6:  # Viral production
            if I >= 1: V += 1
        else:  # Viral clearance
            if V >= 1: V -= 1
            
        # Ensure non-negative and update current state
        T, E, I, R, V = max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)
        current_state[:] = [T, E, I, R, V]
        
        step += 1
    
    # Fill remaining grid points with final state
    while grid_idx < n_grid:
        output_states[grid_idx] = current_state
        grid_idx += 1
    
    return t_grid.copy(), output_states


def _gillespie_streaming_interpolation(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, t_grid, max_steps, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming interpolation: Maintain sliding window of recent states
    for on-the-fly interpolation.
    """
    n_grid = len(t_grid)
    output_states = np.zeros((n_grid, 5))
    
    # Sliding window for interpolation (keep last few points)
    window_size = 10
    time_window = np.zeros(window_size)
    state_window = np.zeros((window_size, 5))
    window_idx = 0
    window_filled = 0
    
    # Initialize
    current_state = np.array([T, E, I, R, V], dtype=float)
    output_states[0] = current_state
    time_window[0] = 0.0
    state_window[0] = current_state
    window_filled = 1
    
    t = 0.0
    step = 0
    grid_idx = 1
    
    while t < t_max and step < max_steps and grid_idx < n_grid:
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate rates (same as sparse)
        a1 = beta * 1e-9 * T * V
        a2 = phi * 1e-5 * T * I
        a3 = rho * R
        a4 = k * E
        a5 = delta * I
        a6 = pi * I
        a7 = c * V
        
        a_total = a1 + a2 + a3 + a4 + a5 + a6 + a7
        if a_total <= 0:
            break
            
        # Sample reaction
        tau = np.random.exponential(1.0 / a_total)
        t_next = t + tau
        
        # Process any grid points between current time and next reaction
        while grid_idx < n_grid and t_grid[grid_idx] <= t_next:
            target_time = t_grid[grid_idx]
            
            # Interpolate state at target_time using window
            if window_filled >= 2:
                # Linear interpolation between last two points
                t1 = time_window[(window_idx - 1) % window_size]
                t2 = time_window[window_idx % window_size] if window_filled > window_idx else t
                s1 = state_window[(window_idx - 1) % window_size]
                s2 = current_state
                
                if t2 > t1:
                    alpha = (target_time - t1) / (t2 - t1)
                    alpha = np.clip(alpha, 0, 1)
                    interpolated_state = s1 + alpha * (s2 - s1)
                else:
                    interpolated_state = s1
                    
                output_states[grid_idx] = interpolated_state
            else:
                output_states[grid_idx] = current_state
                
            grid_idx += 1
            
        if grid_idx >= n_grid:
            break
            
        # Execute reaction and update state (same as sparse)
        t = t_next
        r = np.random.uniform(0, a_total)
        
        if r < a1:
            if T >= 1: T -= 1; E += 1
        elif r < a1 + a2:
            if T >= 1: T -= 1; R += 1
        elif r < a1 + a2 + a3:
            if R >= 1: R -= 1; T += 1
        elif r < a1 + a2 + a3 + a4:
            if E >= 1: E -= 1; I += 1
        elif r < a1 + a2 + a3 + a4 + a5:
            if I >= 1: I -= 1
        elif r < a1 + a2 + a3 + a4 + a5 + a6:
            if I >= 1: V += 1
        else:
            if V >= 1: V -= 1
            
        T, E, I, R, V = max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)
        current_state[:] = [T, E, I, R, V]
        
        # Update sliding window
        window_idx = (window_idx + 1) % window_size
        time_window[window_idx] = t
        state_window[window_idx] = current_state.copy()
        window_filled = min(window_filled + 1, window_size)
        
        step += 1
    
    # Fill remaining grid points
    while grid_idx < n_grid:
        output_states[grid_idx] = current_state
        grid_idx += 1
    
    return t_grid.copy(), output_states


def _gillespie_adaptive_storage(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, t_grid, max_steps, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive storage: Pre-allocate reasonable arrays, resize if needed.
    """
    # Estimate storage needs (heuristic: ~1000 events per day for viral dynamics)
    estimated_events = min(int(t_max * 2000), max_steps // 10)
    initial_size = max(estimated_events, 1000)
    
    # Pre-allocate arrays
    times = np.zeros(initial_size)
    states = np.zeros((initial_size, 5))
    
    # Initialize
    current_state = np.array([T, E, I, R, V], dtype=float)
    times[0] = 0.0
    states[0] = current_state
    
    t = 0.0
    step = 0
    stored_events = 1
    
    while t < t_max and step < max_steps:
        # Check extinction
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Resize arrays if needed
        if stored_events >= len(times):
            new_size = len(times) * 2
            times = np.resize(times, new_size)
            states = np.resize(states, (new_size, 5))
            
        # Calculate rates
        a1 = beta * 1e-9 * T * V
        a2 = phi * 1e-5 * T * I
        a3 = rho * R
        a4 = k * E
        a5 = delta * I
        a6 = pi * I
        a7 = c * V
        
        a_total = a1 + a2 + a3 + a4 + a5 + a6 + a7
        if a_total <= 0:
            break
            
        # Sample and execute reaction
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        
        r = np.random.uniform(0, a_total)
        
        if r < a1:
            if T >= 1: T -= 1; E += 1
        elif r < a1 + a2:
            if T >= 1: T -= 1; R += 1
        elif r < a1 + a2 + a3:
            if R >= 1: R -= 1; T += 1
        elif r < a1 + a2 + a3 + a4:
            if E >= 1: E -= 1; I += 1
        elif r < a1 + a2 + a3 + a4 + a5:
            if I >= 1: I -= 1
        elif r < a1 + a2 + a3 + a4 + a5 + a6:
            if I >= 1: V += 1
        else:
            if V >= 1: V -= 1
            
        T, E, I, R, V = max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)
        current_state[:] = [T, E, I, R, V]
        
        # Store state
        times[stored_events] = t
        states[stored_events] = current_state
        stored_events += 1
        step += 1
    
    # Trim arrays to actual size
    times = times[:stored_events]
    states = states[:stored_events]
    
    # Interpolate to target grid
    if len(times) > 1:
        states_interp = np.zeros((len(t_grid), 5))
        for i in range(5):
            states_interp[:, i] = np.interp(t_grid, times, states[:, i])
        return t_grid.copy(), states_interp
    else:
        return t_grid.copy(), np.tile(states[0], (len(t_grid), 1))


def _gillespie_minimal_storage(
    T, E, I, R, V, beta, pi, delta, phi, rho, k, c,
    t_max, max_steps, extinction_threshold
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal storage: For cases where we don't need interpolation.
    Store only every Nth event or use adaptive sampling.
    """
    # Sample storage - only keep every 100th event for memory efficiency
    sample_interval = 100
    estimated_samples = max_steps // sample_interval + 100
    
    times = np.zeros(estimated_samples)
    states = np.zeros((estimated_samples, 5))
    
    current_state = np.array([T, E, I, R, V], dtype=float)
    times[0] = 0.0
    states[0] = current_state
    
    t = 0.0
    step = 0
    stored_idx = 1
    
    while t < t_max and step < max_steps:
        if (E + I <= extinction_threshold and V <= extinction_threshold):
            break
            
        # Calculate rates
        a1 = beta * 1e-9 * T * V
        a2 = phi * 1e-5 * T * I
        a3 = rho * R
        a4 = k * E
        a5 = delta * I
        a6 = pi * I
        a7 = c * V
        
        a_total = a1 + a2 + a3 + a4 + a5 + a6 + a7
        if a_total <= 0:
            break
            
        tau = np.random.exponential(1.0 / a_total)
        t += tau
        
        r = np.random.uniform(0, a_total)
        
        if r < a1:
            if T >= 1: T -= 1; E += 1
        elif r < a1 + a2:
            if T >= 1: T -= 1; R += 1
        elif r < a1 + a2 + a3:
            if R >= 1: R -= 1; T += 1
        elif r < a1 + a2 + a3 + a4:
            if E >= 1: E -= 1; I += 1
        elif r < a1 + a2 + a3 + a4 + a5:
            if I >= 1: I -= 1
        elif r < a1 + a2 + a3 + a4 + a5 + a6:
            if I >= 1: V += 1
        else:
            if V >= 1: V -= 1
            
        T, E, I, R, V = max(0, T), max(0, E), max(0, I), max(0, R), max(0, V)
        current_state[:] = [T, E, I, R, V]
        
        # Store every sample_interval-th event
        if step % sample_interval == 0 and stored_idx < len(times):
            times[stored_idx] = t
            states[stored_idx] = current_state
            stored_idx += 1
            
        step += 1
    
    return times[:stored_idx], states[:stored_idx]


def benchmark_storage_strategies(n_tests: int = 5):
    """
    Benchmark different storage strategies.
    """
    import time
    from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid
    
    print("Benchmarking TEIRV Storage Strategies")
    print("=" * 50)
    
    strategies = ["sparse", "streaming", "adaptive"]
    results = {}
    
    # Set up test case
    prior = create_teirv_prior()
    ic = get_teirv_initial_conditions()
    t_grid = create_teirv_time_grid(14.0, 1.0)
    
    for strategy in strategies:
        print(f"\nTesting {strategy} storage...")
        times = []
        
        for i in range(n_tests):
            theta = prior.sample().numpy()
            ic['V'] = theta[5]
            
            start = time.time()
            try:
                _, states = gillespie_teirv_optimized(
                    theta, ic, 14.0, t_grid, 
                    storage_strategy=strategy
                )
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Test {i+1}: {elapsed:.3f}s")
                
            except Exception as e:
                print(f"  Test {i+1}: FAILED - {e}")
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[strategy] = {'mean': avg_time, 'std': std_time}
            print(f"  Average: {avg_time:.3f} ± {std_time:.3f}s")
        else:
            results[strategy] = None
    
    # Compare with original
    print(f"\nComparing with original implementation...")
    from teirv_simulator import gillespie_teirv
    
    original_times = []
    for i in range(n_tests):
        theta = prior.sample().numpy()
        ic['V'] = theta[5]
        
        start = time.time()
        try:
            _, states = gillespie_teirv(theta, ic, 14.0, t_grid)
            elapsed = time.time() - start
            original_times.append(elapsed)
        except:
            pass
    
    if original_times:
        orig_avg = np.mean(original_times)
        print(f"Original: {orig_avg:.3f}s average")
        
        print(f"\nSpeedup Summary:")
        print("-" * 30)
        for strategy, result in results.items():
            if result:
                speedup = orig_avg / result['mean']
                print(f"{strategy:>10}: {speedup:.1f}x faster")
    
    return results


if __name__ == '__main__':
    benchmark_storage_strategies()