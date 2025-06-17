#!/usr/bin/env python3
"""
Debug TEIRV performance issues.
"""
import sys
from pathlib import Path
import time
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_simulator import gillespie_teirv
from TEIRV.teirv_utils import create_teirv_prior, get_teirv_initial_conditions, create_teirv_time_grid


def test_single_simulation():
    """Test a single TEIRV simulation."""
    print("Testing single TEIRV simulation performance...")
    
    # Set up simulation
    prior = create_teirv_prior()
    theta = prior.sample().numpy()
    ic = get_teirv_initial_conditions()
    ic['V'] = theta[5]  # Set V0
    t_grid = create_teirv_time_grid(14.0, 1.0)
    
    print(f"Parameters: {theta}")
    print(f"Initial conditions: {ic}")
    print(f"Time grid: {len(t_grid)} points")
    
    # Time the simulation
    start_time = time.time()
    
    try:
        times, states = gillespie_teirv(
            theta=theta,
            initial_conditions=ic,
            t_max=14.0,
            t_grid=t_grid,
            max_steps=100000  # Lower max steps to see if that's the issue
        )
        
        elapsed = time.time() - start_time
        
        print(f"Simulation completed in {elapsed:.2f} seconds")
        print(f"Final state: T={states[-1,0]:.0f}, E={states[-1,1]:.0f}, I={states[-1,2]:.0f}, R={states[-1,3]:.0f}, V={states[-1,4]:.0f}")
        
        return elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Simulation failed after {elapsed:.2f} seconds: {e}")
        return None


def test_multiple_simulations():
    """Test multiple simulations to see if there's a pattern."""
    print("\\nTesting multiple simulations...")
    
    times = []
    for i in range(5):
        print(f"Simulation {i+1}:")
        elapsed = test_single_simulation()
        if elapsed is not None:
            times.append(elapsed)
            print(f"  Time: {elapsed:.2f}s")
        else:
            print(f"  Failed")
    
    if times:
        print(f"\\nAverage time: {np.mean(times):.2f}s")
        print(f"Min/Max time: {np.min(times):.2f}s / {np.max(times):.2f}s")
    else:
        print("No successful simulations")


def test_parameter_sensitivity():
    """Test if certain parameter ranges cause slow simulations."""
    print("\\nTesting parameter sensitivity...")
    
    # Test with different parameter sets
    test_params = [
        [1.0, 250.0, 2.0, 1.0, 0.1, 10.0],     # Low values
        [10.0, 400.0, 6.0, 7.5, 0.5, 50.0],    # Medium values  
        [19.0, 590.0, 10.0, 14.0, 0.9, 140.0], # High values
    ]
    
    for i, theta in enumerate(test_params):
        print(f"\\nParameter set {i+1}: {theta}")
        
        ic = get_teirv_initial_conditions()
        ic['V'] = theta[5]
        t_grid = create_teirv_time_grid(14.0, 1.0)
        
        start_time = time.time()
        
        try:
            times, states = gillespie_teirv(
                theta=theta,
                initial_conditions=ic,
                t_max=14.0,
                t_grid=t_grid,
                max_steps=50000  # Even lower to catch runaway simulations
            )
            
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.2f}s")
            print(f"  Final: T={states[-1,0]:.0f}, V={states[-1,4]:.0f}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  Failed after {elapsed:.2f}s: {e}")


if __name__ == '__main__':
    test_single_simulation()
    #test_multiple_simulations()
    #test_parameter_sensitivity()