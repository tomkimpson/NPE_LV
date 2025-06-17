#!/usr/bin/env python3
"""
Quick test of TEIRV NPE components.
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from teirv_data_generation import TEIRVDataGenerator
from teirv_inference import TEIRVInference


def quick_test():
    print("Quick TEIRV NPE Test")
    print("=" * 40)
    
    # Test 1: Single data generation
    print("1. Testing single sample generation...")
    generator = TEIRVDataGenerator(seed=42)
    theta, x = generator.generate_single()
    print(f"   ✅ Generated: theta={theta.shape}, x={x.shape}")
    print(f"   Parameters: {theta.numpy()}")
    print(f"   Observations: {x.numpy()}")
    
    # Test 2: Small batch generation
    print("\\n2. Testing small batch generation...")
    try:
        theta_batch, x_batch = generator.generate_batch(n_samples=10, batch_size=5)
        print(f"   ✅ Generated batch: theta={theta_batch.shape}, x={x_batch.shape}")
        success_rate = generator.get_stats()['success_rate']
        print(f"   Success rate: {success_rate:.1%}")
    except Exception as e:
        print(f"   ❌ Batch generation failed: {e}")
        return
    
    # Test 3: NPE setup
    print("\\n3. Testing NPE setup...")
    try:
        inference = TEIRVInference(observation_type='rt_pcr')
        inference.setup_inference(x_dim=x_batch.shape[1])
        print(f"   ✅ NPE setup successful")
        print(f"   Observation dimension: {x_batch.shape[1]}")
    except Exception as e:
        print(f"   ❌ NPE setup failed: {e}")
        return
    
    # Test 4: Quick training
    print("\\n4. Testing quick NPE training...")
    try:
        # Use very small dataset and minimal training
        training_info = inference.train(
            theta=theta_batch,
            x=x_batch,
            training_batch_size=5,
            max_num_epochs=3,  # Just a few epochs
            validation_fraction=0.2,
            stop_after_epochs=10
        )
        print(f"   ✅ Training completed")
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return
    
    # Test 5: Inference
    print("\\n5. Testing inference...")
    try:
        # Use first observation as test
        x_test = x_batch[0]
        posterior_samples = inference.sample_posterior(x_test, num_samples=100)
        print(f"   ✅ Inference successful: {posterior_samples.shape}")
        
        # Quick summary
        samples_mean = posterior_samples.mean(dim=0)
        true_params = theta_batch[0]
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        
        print("   Parameter recovery:")
        for i, name in enumerate(param_names):
            print(f"     {name}: true={true_params[i]:.2f}, pred={samples_mean[i]:.2f}")
            
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return
    
    print("\\n✅ All tests passed! TEIRV NPE pipeline is working.")


if __name__ == '__main__':
    quick_test()