#!/usr/bin/env python3
"""
Test script for TEIRV simulator implementation.
Validates the Gillespie algorithm and observation model.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from teirv_simulator import gillespie_teirv, simulate_teirv_batch, check_teirv_trajectory_validity
from teirv_utils import (create_teirv_prior, get_teirv_initial_conditions, 
                        create_teirv_time_grid, apply_observation_model,
                        visualize_teirv_trajectory, get_teirv_fixed_parameters)


def test_single_simulation():
    """Test single TEIRV simulation."""
    print("Testing single TEIRV simulation...")
    
    # Sample parameters from prior
    prior = create_teirv_prior()
    theta = prior.sample().numpy()
    
    print(f"Sampled parameters: {theta}")
    
    # Set up simulation
    t_max = 14.0
    t_grid = create_teirv_time_grid(t_max)
    ic = get_teirv_initial_conditions(V0=theta[5])  # Use sampled V0
    
    print(f"Initial conditions: {ic}")
    print(f"Time grid: {len(t_grid)} points from 0 to {t_max}")
    
    # Run simulation
    try:
        times, states = gillespie_teirv(theta, ic, t_max, t_grid)
        
        print(f"Simulation successful!")
        print(f"Final state: T={states[-1,0]:.0f}, E={states[-1,1]:.0f}, "
              f"I={states[-1,2]:.0f}, R={states[-1,3]:.0f}, V={states[-1,4]:.0f}")
        
        # Check validity
        is_valid = check_teirv_trajectory_validity(states)
        print(f"Trajectory valid: {is_valid}")
        
        return times, states, theta
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None, None


def test_observation_model():
    """Test RT-PCR observation model."""
    print("\nTesting observation model...")
    
    # Create synthetic viral load trajectory
    t_grid = create_teirv_time_grid()
    
    # Simple exponential growth then decay
    V_true = 1000 * np.exp(2 * t_grid) * np.exp(-0.5 * t_grid)
    
    # Apply observation model
    observations = apply_observation_model(V_true, sigma=1.0, add_noise=True, seed=42)
    
    print(f"True viral loads (first 5): {V_true[:5]}")
    print(f"Observations (first 5): {observations[:5]}")
    print(f"Log10 true VL (first 5): {np.log10(V_true[:5])}")
    
    # Check detection limit handling
    low_vl = np.array([1e-3, 1e-1, 1e1, 1e3])
    obs_low = apply_observation_model(low_vl, sigma=0.1, detection_limit=-0.65)
    print(f"Low VL test - True: {low_vl}, Observed: {obs_low}")


def test_batch_simulation():
    """Test batch simulation."""
    print("\nTesting batch simulation...")
    
    # Sample multiple parameter sets
    prior = create_teirv_prior()
    n_batch = 5
    theta_batch = prior.sample((n_batch,)).numpy()
    
    print(f"Batch size: {n_batch}")
    
    # Set up simulation
    t_max = 14.0
    t_grid = create_teirv_time_grid(t_max)
    ic = get_teirv_initial_conditions()  # V0 will be overridden
    
    # Run batch simulation
    try:
        trajectories = simulate_teirv_batch(theta_batch, ic, t_max, t_grid)
        
        print(f"Batch simulation successful!")
        print(f"Output shape: {trajectories.shape}")
        
        # Check validity of each trajectory
        valid_count = 0
        for i in range(n_batch):
            if check_teirv_trajectory_validity(trajectories[i]):
                valid_count += 1
        
        print(f"Valid trajectories: {valid_count}/{n_batch}")
        
        return trajectories, theta_batch
        
    except Exception as e:
        print(f"Batch simulation failed: {e}")
        return None, None


def test_clinical_data_format():
    """Test loading and processing clinical data format."""
    print("\nTesting clinical data format...")
    
    # Load example patient data
    data_path = Path(__file__).parent.parent / "external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/432192.ssv"
    
    if data_path.exists():
        # Read the data
        data = np.loadtxt(data_path, skiprows=1)
        times = data[:, 0]
        observations = data[:, 1]
        
        print(f"Patient 432192 data:")
        print(f"Time points: {times}")
        print(f"Observations: {observations}")
        print(f"Detection limit (-0.65) count: {np.sum(observations == -0.65)}")
        
        return times, observations
    else:
        print(f"Clinical data not found at {data_path}")
        return None, None


def test_parameter_ranges():
    """Test that parameter samples are in reasonable ranges."""
    print("\nTesting parameter ranges...")
    
    prior = create_teirv_prior()
    samples = prior.sample((1000,)).numpy()
    
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    expected_ranges = [
        (1e-12, 1e-6),   # β
        (10, 1000),      # π
        (0.1, 5),        # δ
        (1e-12, 1e-6),   # φ
        (0.01, 1),       # ρ
        (100, 10000)     # V₀
    ]
    
    for i, (name, (low, high)) in enumerate(zip(param_names, expected_ranges)):
        param_samples = samples[:, i]
        mean_val = np.mean(param_samples)
        in_range = np.all((param_samples >= low) & (param_samples <= high))
        
        print(f"{name}: mean={mean_val:.2e}, range_ok={in_range}")
        print(f"  Sample range: [{np.min(param_samples):.2e}, {np.max(param_samples):.2e}]")


def main():
    """Run all tests."""
    print("="*60)
    print("TEIRV SIMULATOR VALIDATION TESTS")
    print("="*60)
    
    # Test 1: Single simulation
    times, states, theta = test_single_simulation()
    
    # Test 2: Observation model
    test_observation_model()
    
    # Test 3: Batch simulation
    trajectories, theta_batch = test_batch_simulation()
    
    # Test 4: Clinical data format
    clinical_times, clinical_obs = test_clinical_data_format()
    
    # Test 5: Parameter ranges
    test_parameter_ranges()
    
    # Visualization if single simulation worked
    if times is not None and states is not None:
        print("\nCreating visualization...")
        
        # Apply observation model to simulated data
        observations = apply_observation_model(states[:, 4], sigma=1.0, seed=42)
        
        # Create plot
        fig = visualize_teirv_trajectory(times, states, observations, theta)
        
        # Save plot
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "teirv_test_simulation.png", dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_dir / 'teirv_test_simulation.png'}")
        
        plt.close(fig)
    
    print("\n" + "="*60)
    print("TEIRV VALIDATION TESTS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()