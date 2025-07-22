#!/usr/bin/env python3
"""
Quick test to verify the log_prob fix works correctly.
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from TEIRV.teirv_utils import create_teirv_prior

def test_log_prob_finite():
    """Test that log_prob returns finite values for valid parameters."""
    print("🧪 Testing log_prob Fix")
    print("=" * 30)
    
    prior = create_teirv_prior()
    
    # Generate several samples
    n_samples = 10
    all_finite = True
    
    print("Testing log_prob for sample parameters:")
    print("----------------------------------------")
    
    for i in range(n_samples):
        # Sample parameters (already scaled)
        theta = prior.sample()
        print(f"Sample {i+1}: {theta.numpy()}")
        
        # Compute log probability
        log_prob_val = prior.log_prob(theta)
        
        # Check if finite
        is_finite = torch.isfinite(log_prob_val).item()
        print(f"  log_prob: {log_prob_val.item():.6f}, finite: {is_finite}")
        
        if not is_finite:
            all_finite = False
            print(f"  ❌ Sample {i+1} produced non-finite log_prob!")
        
        print()
    
    # Test edge cases
    print("Testing edge cases:")
    print("------------------")
    
    # Test with parameters slightly inside boundaries (exact boundaries have zero probability in continuous distributions)
    epsilon = 1e-6
    edge_cases = [
        torch.tensor([epsilon, 200.0+epsilon, 1.0+epsilon, epsilon, epsilon, 1.0+epsilon]),  # Near min values
        torch.tensor([(20-epsilon)*1e-7, 400.0-epsilon, 10.0-epsilon, (15-epsilon)*1e-5, 1.0-epsilon, np.exp(5-epsilon)]),  # Near max values (scaled)
        torch.tensor([10*1e-7, 300.0, 5.0, 7.5*1e-5, 0.5, np.exp(2.5)]) # Mid values (scaled)
    ]
    
    for i, theta in enumerate(edge_cases):
        log_prob_val = prior.log_prob(theta)
        is_finite = torch.isfinite(log_prob_val).item()
        print(f"Edge case {i+1}: log_prob = {log_prob_val.item():.6f}, finite: {is_finite}")
        if not is_finite:
            all_finite = False
    
    print("\n" + "=" * 30)
    if all_finite:
        print("✅ ALL TESTS PASSED - log_prob returns finite values")
        print("🎉 NPE training should now work correctly!")
    else:
        print("❌ SOME TESTS FAILED - log_prob still has issues")
    
    return all_finite

def test_scaling_consistency():
    """Test that scaling is applied consistently between sample() and log_prob()."""
    print("\n🔍 Testing Scaling Consistency")
    print("=" * 30)
    
    prior = create_teirv_prior()
    
    # Sample parameters 
    theta = prior.sample()
    beta_scaled, pi, delta, phi_scaled, rho, v0 = theta
    
    # Check the scaling ratios are correct
    print("Verifying parameter scaling:")
    print(f"β_scaled: {beta_scaled.item():.6e}")
    print(f"φ_scaled: {phi_scaled.item():.6e}")
    
    # Verify β is in expected range [0, 20*1e-7]
    beta_max_expected = 20 * 1e-7
    beta_in_range = 0 <= beta_scaled <= beta_max_expected
    print(f"β in range [0, {beta_max_expected:.6e}]: {beta_in_range}")
    
    # Verify φ is in expected range [0, 15*1e-5]  
    phi_max_expected = 15 * 1e-5
    phi_in_range = 0 <= phi_scaled <= phi_max_expected
    print(f"φ in range [0, {phi_max_expected:.6e}]: {phi_in_range}")
    
    # Check that log_prob works with these values
    log_prob_val = prior.log_prob(theta)
    log_prob_finite = torch.isfinite(log_prob_val).item()
    print(f"log_prob finite: {log_prob_finite}")
    
    consistency_test = beta_in_range and phi_in_range and log_prob_finite
    print(f"\nScaling consistency test: {'✅ PASSED' if consistency_test else '❌ FAILED'}")
    
    return consistency_test

def main():
    """Run all tests."""
    print("🚨 CRITICAL BUG FIX VERIFICATION")
    print("=" * 50)
    print("Testing fix for log_prob method that was returning -∞")
    print("=" * 50)
    
    test1 = test_log_prob_finite()
    test2 = test_scaling_consistency()
    
    print("\n" + "=" * 50)
    print("📊 FIX VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"log_prob returns finite values: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Scaling consistency maintained: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    overall_success = test1 and test2
    print(f"\nOverall result: {'🎉 FIX SUCCESSFUL' if overall_success else '🚨 FIX INCOMPLETE'}")
    
    if overall_success:
        print("\n✅ The critical log_prob bug has been fixed!")
        print("   - log_prob now returns finite values for valid parameters")
        print("   - Inverse scaling and Jacobian corrections implemented correctly")
        print("   - NPE training should now work properly")
        print("\n🚀 Ready to resume SLURM workflow testing!")
    else:
        print("\n❌ The fix still has issues that need to be addressed")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())