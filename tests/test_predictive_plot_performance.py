#!/usr/bin/env python3
"""
Test script to isolate and debug predictive plot generation performance.
Loads existing inference results and recreates just the predictive plot step.
"""

import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_simulator import gillespie_teirv
from TEIRV.teirv_utils import get_teirv_initial_conditions, apply_observation_model

def create_predictive_plot_test(patient_id: str, posterior_samples: torch.Tensor, 
                               observations: torch.Tensor, n_pred_samples: int = 100) -> plt.Figure:
    """
    Test version of _create_predictive_plot with timing information.
    """
    print(f"=== Testing predictive plot generation ===")
    print(f"Patient: {patient_id}")
    print(f"Posterior samples shape: {posterior_samples.shape}")
    print(f"Observations shape: {observations.shape}")
    print(f"Number of prediction samples: {n_pred_samples}")
    
    start_time = time.time()
    
    # Time grids
    t_obs = np.arange(0, 15, 1.0)  # Observed range: 0-14 days
    t_pred = np.arange(0, 21, 1.0)  # Extended range: 0-20 days
    
    # Select subset of posterior samples
    n_samples = min(n_pred_samples, len(posterior_samples))
    sample_indices = np.random.choice(len(posterior_samples), n_samples, replace=False)
    selected_samples = posterior_samples[sample_indices]
    
    print(f"Selected {n_samples} posterior samples")
    
    # Generate predictions
    predictions_obs = []  # For observed range
    predictions_ext = []  # For extended range
    
    base_ic = get_teirv_initial_conditions()
    
    sim_times = []
    
    for i, theta_sample in enumerate(selected_samples):
        sim_start = time.time()
        
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{n_samples}...")
        
        try:
            # Set up initial conditions
            ic = base_ic.copy()
            ic['V'] = theta_sample[5].item()  # V₀ from posterior
            
            # Simulate for extended range
            _, trajectory_ext = gillespie_teirv(
                theta=theta_sample.numpy(),
                initial_conditions=ic,
                t_max=20.0,
                t_grid=t_pred,
                max_steps=1000000
            )
            
            # Apply observation model (RT-PCR transformation)
            V_trajectory_ext = trajectory_ext[:, 4]  # V compartment
            obs_ext = apply_observation_model(
                V_trajectory=V_trajectory_ext,
                sigma=1.0,  # Standard observation noise
                detection_limit=-0.65,
                add_noise=True
            )
            
            # Extract observed range (first 15 points: 0-14 days)
            obs_range = obs_ext[:15]
            
            predictions_obs.append(obs_range)
            predictions_ext.append(obs_ext)
            
            sim_time = time.time() - sim_start
            sim_times.append(sim_time)
            
        except Exception as e:
            print(f"    Simulation {i} failed: {e}")
            continue
    
    avg_sim_time = np.mean(sim_times) if sim_times else 0
    total_sim_time = time.time() - start_time
    
    print(f"Simulation timing:")
    print(f"  Average per simulation: {avg_sim_time:.3f}s")
    print(f"  Total simulation time: {total_sim_time:.3f}s")
    print(f"  Successful simulations: {len(predictions_obs)}/{n_samples}")
    
    if len(predictions_obs) == 0:
        print("  ❌ No successful predictions generated")
        return plt.figure()
    
    # Convert to arrays and compute credible intervals
    pred_obs_array = np.array(predictions_obs)  # Shape: (n_samples, 15)
    pred_ext_array = np.array(predictions_ext)  # Shape: (n_samples, 21)
    
    print(f"Computing credible intervals...")
    
    # Compute quantiles
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.95]
    obs_quantiles = np.quantile(pred_obs_array, quantiles, axis=0)  # Shape: (5, 15)
    ext_quantiles = np.quantile(pred_ext_array, quantiles, axis=0)  # Shape: (5, 21)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot observed data
    obs_data = observations.cpu().numpy()
    ax.scatter(t_obs, obs_data, color='black', s=50, zorder=10, label='Observed data')
    
    # Plot credible intervals for observed range (red)
    ax.fill_between(t_obs, obs_quantiles[0], obs_quantiles[4], 
                    alpha=0.2, color='red', label='95% CI (observed range)')
    ax.fill_between(t_obs, obs_quantiles[1], obs_quantiles[3], 
                    alpha=0.4, color='red', label='50% CI (observed range)')
    ax.plot(t_obs, obs_quantiles[2], color='red', linewidth=2, label='Median (observed range)')
    
    # Get extended range excluding observed range
    t_pred_only = t_pred[15:]  # Days 15-20
    ext_quantiles_pred = ext_quantiles[:, 15:]  # Shape: (5, 6)
    
    # Plot credible intervals for prediction range (purple)
    ax.fill_between(t_pred_only, ext_quantiles_pred[0], ext_quantiles_pred[4], 
                    alpha=0.2, color='purple', label='95% CI (predicted range)')
    ax.fill_between(t_pred_only, ext_quantiles_pred[1], ext_quantiles_pred[3], 
                    alpha=0.4, color='purple', label='50% CI (predicted range)')
    ax.plot(t_pred_only, ext_quantiles_pred[2], color='purple', linewidth=2, 
            label='Median (predicted range)')
    
    # Add boundary line
    ax.axvline(x=14, color='gray', linestyle='--', alpha=0.7, 
               label='Observed/Predicted boundary')
    
    # Formatting
    ax.set_xlabel('Days')
    ax.set_ylabel('Viral Load (log10 copies/mL)')
    ax.set_yscale('log')
    ax.set_title(f'Patient {patient_id}: Predictive Plot with Credible Intervals')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.3f}s")
    
    plt.tight_layout()
    return fig

def main():
    """Test predictive plot generation on a single patient."""
    
    # Load data for patient 443108 (the one that was hanging)
    patient_id = "443108"
    results_dir = Path("/fred/oz022/tkimpson/NPE_LV/workflows/demo/inference_results")
    patient_dir = results_dir / f"patient_{patient_id}"
    
    print(f"Loading data for patient {patient_id}...")
    
    # Load posterior samples and observations
    posterior_samples = torch.from_numpy(np.load(patient_dir / "posterior_samples.npy"))
    observations = torch.from_numpy(np.load(patient_dir / "observations.npy"))
    
    print(f"Loaded:")
    print(f"  Posterior samples: {posterior_samples.shape}")
    print(f"  Observations: {observations.shape}")
    
    # Test with different numbers of samples
    test_samples = [5, 10, 20, 50, 100]
    
    for n_samples in test_samples:
        print(f"\n{'='*60}")
        print(f"Testing with {n_samples} posterior samples")
        print('='*60)
        
        fig = create_predictive_plot_test(patient_id, posterior_samples, observations, n_samples)
        
        # Save test plot
        output_path = Path(__file__).parent / "test_output" / f"predictive_test_{n_samples}samples.png"
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Test plot saved: {output_path}")

if __name__ == "__main__":
    main()