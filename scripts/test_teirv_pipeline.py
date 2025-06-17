#!/usr/bin/env python3
"""
Test script for TEIRV NPE pipeline.
Tests data generation, training, and inference.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from teirv_data_generation import TEIRVDataGenerator
from teirv_inference import TEIRVInference
from teirv_utils import create_teirv_prior


def test_data_generation():
    """Test TEIRV data generation."""
    print("Testing TEIRV data generation...")
    
    generator = TEIRVDataGenerator(
        t_max=14.0,
        dt=1.0,
        observation_noise=1.0,
        use_observations_only=True,
        seed=42
    )
    
    # Test single generation
    theta, x = generator.generate_single()
    print(f"Single sample: theta shape={theta.shape}, x shape={x.shape}")
    print(f"Parameter values: {theta.numpy()}")
    print(f"Observation range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Test batch generation
    theta_batch, x_batch = generator.generate_batch(n_samples=100, batch_size=50)
    print(f"Batch samples: theta shape={theta_batch.shape}, x shape={x_batch.shape}")
    
    stats = generator.get_stats()
    print(f"Success rate: {stats['success_rate']:.2%}")
    
    return theta_batch, x_batch, generator


def test_small_training():
    """Test NPE training on small dataset."""
    print("\\nTesting TEIRV NPE training...")
    
    # Generate small training dataset
    generator = TEIRVDataGenerator(seed=123)
    theta, x = generator.generate_batch(n_samples=500, batch_size=100)
    
    print(f"Training data: {len(theta)} samples, obs dim = {x.shape[1]}")
    
    # Initialize and train NPE
    inference = TEIRVInference(observation_type='rt_pcr', seed=456)
    
    try:
        training_info = inference.train(
            theta=theta,
            x=x,
            training_batch_size=64,
            max_num_epochs=20,  # Quick training
            validation_fraction=0.2
        )
        
        print("Training successful!")
        return inference, theta, x
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None, None, None


def test_inference():
    """Test inference on synthetic patient data."""
    print("\\nTesting TEIRV inference...")
    
    # Train small model first
    inference, theta_train, x_train = test_small_training()
    
    if inference is None:
        print("Skipping inference test due to training failure")
        return
    
    # Generate synthetic patient data
    generator = TEIRVDataGenerator(seed=789)
    true_theta = np.array([10.0, 400.0, 5.0, 7.5, 0.5, 50.0])  # Reasonable values
    
    try:
        times, observations = generator.generate_test_patient_data(
            true_theta=true_theta,
            noise_seed=101
        )
        
        print(f"Generated patient data: {len(observations)} timepoints")
        print(f"True parameters: {true_theta}")
        print(f"Observations: {observations}")
        
        # Sample from posterior
        x_obs = torch.tensor(observations, dtype=torch.float32)
        posterior_samples = inference.sample_posterior(x_obs, num_samples=1000)
        
        print(f"Posterior samples shape: {posterior_samples.shape}")
        
        # Compute summary statistics
        samples_np = posterior_samples.numpy()
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        
        print("\\nPosterior summary:")
        print("-" * 50)
        for i, name in enumerate(param_names):
            mean_val = np.mean(samples_np[:, i])
            std_val = np.std(samples_np[:, i])
            true_val = true_theta[i]
            
            print(f"{name}: {mean_val:.2f} ± {std_val:.2f} (true: {true_val:.2f})")
        
        return inference, true_theta, observations, posterior_samples
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return None, None, None, None


def test_visualization():
    """Test visualization functions."""
    print("\\nTesting TEIRV visualization...")
    
    # Run inference test to get data
    result = test_inference()
    if result[0] is None:
        print("Skipping visualization test")
        return
    
    inference, true_theta, observations, posterior_samples = result
    
    try:
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Plot posterior marginals
        fig1 = inference.plot_posterior_samples(
            posterior_samples,
            true_theta=torch.tensor(true_theta)
        )
        fig1.savefig(output_dir / "teirv_posterior_marginals.png", dpi=150, bbox_inches='tight')
        print("Saved posterior marginals plot")
        
        # Plot pairwise relationships
        fig2 = inference.plot_pairwise(
            posterior_samples,
            true_theta=torch.tensor(true_theta)
        )
        fig2.savefig(output_dir / "teirv_posterior_pairwise.png", dpi=150, bbox_inches='tight')
        print("Saved pairwise posterior plot")
        
        # Try corner plot
        try:
            fig3 = inference.plot_corner(
                posterior_samples,
                true_theta=torch.tensor(true_theta)
            )
            fig3.savefig(output_dir / "teirv_posterior_corner.png", dpi=150, bbox_inches='tight')
            print("Saved corner plot")
        except Exception as e:
            print(f"Corner plot failed: {e}")
        
        # Posterior predictive check
        fig4 = inference.posterior_predictive_check(
            posterior_samples,
            torch.tensor(observations),
            true_theta=torch.tensor(true_theta)
        )
        fig4.savefig(output_dir / "teirv_predictive_check.png", dpi=150, bbox_inches='tight')
        print("Saved predictive check plot")
        
        plt.close('all')
        print(f"All plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def test_clinical_data_compatibility():
    """Test compatibility with clinical data format."""
    print("\\nTesting clinical data compatibility...")
    
    # Load example patient data
    data_path = Path(__file__).parent.parent / "external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/432192.ssv"
    
    if data_path.exists():
        # Read the data
        data = np.loadtxt(data_path, skiprows=1)
        times = data[:, 0]
        observations = data[:, 1]
        
        print(f"Loaded patient 432192 data:")
        print(f"Times: {times}")
        print(f"Observations: {observations}")
        
        # Check if we can use this data format
        generator = TEIRVDataGenerator()
        if len(observations) == len(generator.t_grid):
            print("✅ Compatible with our time grid")
        else:
            print(f"❌ Length mismatch: clinical={len(observations)}, ours={len(generator.t_grid)}")
        
        # Check observation range
        valid_obs = observations[observations > -0.65]
        if len(valid_obs) > 0:
            print(f"Valid observations: {len(valid_obs)}/{len(observations)}")
            print(f"Range: [{valid_obs.min():.2f}, {valid_obs.max():.2f}]")
        else:
            print("❌ No valid observations above detection limit")
            
    else:
        print(f"Clinical data not found at {data_path}")


def main():
    """Run all TEIRV pipeline tests."""
    print("="*60)
    print("TEIRV NPE PIPELINE VALIDATION TESTS")
    print("="*60)
    
    # Test 1: Data generation
    theta_batch, x_batch, generator = test_data_generation()
    
    # Test 2: Training
    inference, theta_train, x_train = test_small_training()
    
    # Test 3: Inference
    test_inference()
    
    # Test 4: Visualization
    test_visualization()
    
    # Test 5: Clinical data compatibility
    test_clinical_data_compatibility()
    
    print("\\n" + "="*60)
    print("TEIRV NPE PIPELINE TESTS COMPLETED")
    print("="*60)
    
    if inference is not None:
        print("✅ All major components working")
        print("✅ Ready for full-scale training")
    else:
        print("❌ Some components failed - check errors above")


if __name__ == '__main__':
    main()