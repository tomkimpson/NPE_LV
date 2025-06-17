#!/usr/bin/env python3
"""
Debug script to analyze TEIRV batch generation performance.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import numpy as np
from teirv_data_generation import TEIRVDataGenerator
from teirv_simulator import simulate_teirv_batch, gillespie_teirv
from teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid, apply_observation_model

def time_single_simulation():
    """Time a single simulation."""
    print("=== Timing Single Simulation ===")
    
    generator = TEIRVDataGenerator(seed=42)
    
    # Time multiple single simulations
    times = []
    for i in range(10):
        start = time.time()
        theta, x = generator.generate_single()
        end = time.time()
        times.append(end - start)
        
    avg_time = np.mean(times)
    print(f"Average single simulation time: {avg_time:.3f}s")
    print(f"Expected time for 5 simulations: {5 * avg_time:.3f}s")
    return avg_time

def time_batch_simulation():
    """Time batch simulation."""
    print("\n=== Timing Batch Simulation ===")
    
    generator = TEIRVDataGenerator(seed=42)
    
    start = time.time()
    theta_batch, x_batch = generator.generate_batch(5)
    end = time.time()
    
    batch_time = end - start
    print(f"Batch simulation time (5 samples): {batch_time:.3f}s")
    print(f"Time per sample in batch: {batch_time/5:.3f}s")
    return batch_time

def analyze_batch_components():
    """Break down batch generation to find bottlenecks."""
    print("\n=== Analyzing Batch Components ===")
    
    generator = TEIRVDataGenerator(seed=42)
    n_samples = 5
    
    # Time parameter sampling
    start = time.time()
    theta_batch = generator.prior.sample((n_samples,)).numpy()
    param_time = time.time() - start
    print(f"Parameter sampling time: {param_time:.4f}s")
    
    # Time initial condition setup
    start = time.time()
    ic_batch = []
    for j in range(n_samples):
        ic = generator.base_ic.copy()
        ic['V'] = theta_batch[j, 5]
        ic_batch.append(ic)
    ic_time = time.time() - start
    print(f"Initial condition setup time: {ic_time:.4f}s")
    
    # Time batch simulation
    start = time.time()
    trajectories = simulate_teirv_batch(
        theta_batch=theta_batch,
        initial_conditions=generator.base_ic,
        t_max=generator.t_max,
        t_grid=generator.t_grid
    )
    sim_time = time.time() - start
    print(f"Batch simulation time: {sim_time:.3f}s")
    
    # Time observation model application
    start = time.time()
    observations = []
    for j in range(n_samples):
        trajectory = trajectories[j]
        V_trajectory = trajectory[:, 4]
        obs = apply_observation_model(
            V_trajectory=V_trajectory,
            sigma=generator.observation_noise,
            detection_limit=generator.detection_limit,
            add_noise=True
        )
        observations.append(obs)
    obs_time = time.time() - start
    print(f"Observation model time: {obs_time:.4f}s")
    
    total_component_time = param_time + ic_time + sim_time + obs_time
    print(f"Total component time: {total_component_time:.3f}s")
    
    return {
        'param_time': param_time,
        'ic_time': ic_time, 
        'sim_time': sim_time,
        'obs_time': obs_time,
        'total': total_component_time
    }

def compare_individual_vs_batch_simulation():
    """Compare individual simulations vs batch simulation."""
    print("\n=== Comparing Individual vs Batch Simulation ===")
    
    generator = TEIRVDataGenerator(seed=42)
    n_samples = 5
    
    # Sample same parameters for fair comparison
    theta_batch = generator.prior.sample((n_samples,)).numpy()
    
    # Time individual simulations
    start = time.time()
    individual_results = []
    for i in range(n_samples):
        ic = generator.base_ic.copy()
        ic['V'] = theta_batch[i, 5]
        
        _, trajectory = gillespie_teirv(
            theta=theta_batch[i],
            initial_conditions=ic,
            t_max=generator.t_max,
            t_grid=generator.t_grid
        )
        
        V_trajectory = trajectory[:, 4]
        observations = apply_observation_model(
            V_trajectory=V_trajectory,
            sigma=generator.observation_noise,
            detection_limit=generator.detection_limit,
            add_noise=True
        )
        individual_results.append((theta_batch[i], observations))
    individual_time = time.time() - start
    
    # Time batch simulation
    start = time.time()
    trajectories = simulate_teirv_batch(
        theta_batch=theta_batch,
        initial_conditions=generator.base_ic,
        t_max=generator.t_max,
        t_grid=generator.t_grid
    )
    
    batch_results = []
    for j in range(n_samples):
        trajectory = trajectories[j]
        V_trajectory = trajectory[:, 4]
        observations = apply_observation_model(
            V_trajectory=V_trajectory,
            sigma=generator.observation_noise,
            detection_limit=generator.detection_limit,
            add_noise=True
        )
        batch_results.append((theta_batch[j], observations))
    batch_time = time.time() - start
    
    print(f"Individual simulations time: {individual_time:.3f}s")
    print(f"Batch simulation time: {batch_time:.3f}s")
    print(f"Speedup factor: {individual_time/batch_time:.2f}x")
    
    return individual_time, batch_time

def profile_gillespie_calls():
    """Profile the Gillespie algorithm calls in batch processing."""
    print("\n=== Profiling Gillespie Calls ===")
    
    generator = TEIRVDataGenerator(seed=42)
    n_samples = 5
    theta_batch = generator.prior.sample((n_samples,)).numpy()
    
    # Time each individual Gillespie call in the batch
    gillespie_times = []
    for i in range(n_samples):
        ic = generator.base_ic.copy()
        ic['V'] = theta_batch[i, 5]
        
        start = time.time()
        _, trajectory = gillespie_teirv(
            theta=theta_batch[i],
            initial_conditions=ic,
            t_max=generator.t_max,
            t_grid=generator.t_grid
        )
        end = time.time()
        gillespie_times.append(end - start)
        print(f"Gillespie simulation {i+1}: {end-start:.3f}s")
    
    print(f"Total Gillespie time: {sum(gillespie_times):.3f}s")
    print(f"Average Gillespie time: {np.mean(gillespie_times):.3f}s")
    
    return gillespie_times

if __name__ == "__main__":
    print("TEIRV Batch Performance Analysis")
    print("=" * 50)
    
    # Run timing tests
    single_time = time_single_simulation()
    batch_time = time_batch_simulation()
    
    print(f"\n=== Summary ===")
    print(f"Single simulation time: {single_time:.3f}s")
    print(f"Batch simulation time: {batch_time:.3f}s")
    print(f"Expected vs actual ratio: {batch_time/(5*single_time):.2f}x")
    
    # Analyze components
    components = analyze_batch_components()
    
    # Compare individual vs batch
    ind_time, batch_time_2 = compare_individual_vs_batch_simulation()
    
    # Profile Gillespie calls
    gillespie_times = profile_gillespie_calls()
    
    print(f"\n=== Final Analysis ===")
    print(f"Bottleneck appears to be: {'Simulation' if components['sim_time'] > 0.8 * components['total'] else 'Other components'}")
    print(f"Simulation takes {components['sim_time']/components['total']*100:.1f}% of total time")