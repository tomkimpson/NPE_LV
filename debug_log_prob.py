#!/usr/bin/env python3
"""
Debug the log_prob issue with edge cases.
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from TEIRV.teirv_utils import create_teirv_prior

def debug_edge_case():
    """Debug the failing edge case in detail."""
    print("🔍 Debugging Edge Case 2 (Max Values)")
    print("=" * 40)
    
    prior = create_teirv_prior()
    
    # The problematic edge case: max values
    theta = torch.tensor([20*1e-7, 400.0, 10.0, 15*1e-5, 1.0, np.exp(5)])
    print(f"Input theta: {theta}")
    
    # Extract scaled parameters
    beta_scaled, pi, delta, phi_scaled, rho, v0 = theta
    
    print(f"\nScaled parameters:")
    print(f"  beta_scaled: {beta_scaled.item():.6e}")
    print(f"  phi_scaled:  {phi_scaled.item():.6e}")
    print(f"  pi:          {pi.item():.1f}")
    print(f"  delta:       {delta.item():.1f}")
    print(f"  rho:         {rho.item():.1f}")
    print(f"  v0:          {v0.item():.1f}")
    
    # Un-scale parameters
    beta_unscaled = beta_scaled / 1e-7
    phi_unscaled = phi_scaled / 1e-5
    
    print(f"\nUnscaled parameters:")
    print(f"  beta_unscaled: {beta_unscaled.item():.6f}")
    print(f"  phi_unscaled:  {phi_unscaled.item():.6f}")
    
    # Check bounds
    print(f"\nParameter bounds:")
    print(f"  beta bounds: {prior.beta_bounds}")
    print(f"  pi bounds:   {prior.pi_bounds}")
    print(f"  delta bounds: {prior.delta_bounds}")
    print(f"  phi bounds:  {prior.phi_bounds}")
    print(f"  rho bounds:  {prior.rho_bounds}")
    print(f"  lnv0 bounds: {prior.lnv0_bounds}")
    
    # Check each parameter's log_prob individually
    print(f"\nIndividual log probabilities:")
    
    try:
        beta_log_prob = prior.beta_dist.log_prob(beta_unscaled)
        print(f"  beta:  {beta_log_prob.item():.6f}")
    except Exception as e:
        print(f"  beta:  ERROR - {e}")
    
    try:
        pi_log_prob = prior.pi_dist.log_prob(pi)
        print(f"  pi:    {pi_log_prob.item():.6f}")
    except Exception as e:
        print(f"  pi:    ERROR - {e}")
    
    try:
        delta_log_prob = prior.delta_dist.log_prob(delta)
        print(f"  delta: {delta_log_prob.item():.6f}")
    except Exception as e:
        print(f"  delta: ERROR - {e}")
    
    try:
        phi_log_prob = prior.phi_dist.log_prob(phi_unscaled)
        print(f"  phi:   {phi_log_prob.item():.6f}")
    except Exception as e:
        print(f"  phi:   ERROR - {e}")
    
    try:
        rho_log_prob = prior.rho_dist.log_prob(rho)
        print(f"  rho:   {rho_log_prob.item():.6f}")
    except Exception as e:
        print(f"  rho:   ERROR - {e}")
    
    try:
        lnv0 = torch.log(v0)
        lnv0_log_prob = prior.lnv0_dist.log_prob(lnv0)
        v0_jacobian = -lnv0
        print(f"  lnv0:  {lnv0_log_prob.item():.6f}")
        print(f"  v0_jac: {v0_jacobian.item():.6f}")
    except Exception as e:
        print(f"  v0:    ERROR - {e}")
    
    # Check Jacobian corrections
    print(f"\nJacobian corrections:")
    beta_jacobian = -torch.log(torch.tensor(1e-7))
    phi_jacobian = -torch.log(torch.tensor(1e-5))
    print(f"  beta_jacobian: {beta_jacobian.item():.6f}")
    print(f"  phi_jacobian:  {phi_jacobian.item():.6f}")
    
    # Try the full log_prob
    print(f"\nFull log_prob:")
    try:
        full_log_prob = prior.log_prob(theta)
        print(f"  Result: {full_log_prob.item():.6f}")
        print(f"  Finite: {torch.isfinite(full_log_prob).item()}")
    except Exception as e:
        print(f"  ERROR: {e}")

def test_boundary_precision():
    """Test if the issue is numerical precision at boundaries."""
    print("\n🔬 Testing Boundary Precision Issues")
    print("=" * 40)
    
    prior = create_teirv_prior()
    
    # Test values slightly inside boundaries
    epsilon = 1e-10
    
    test_cases = [
        "exact boundaries",
        "slightly inside boundaries"
    ]
    
    test_values = [
        # Exact boundaries
        torch.tensor([20*1e-7, 400.0, 10.0, 15*1e-5, 1.0, np.exp(5)]),
        # Slightly inside
        torch.tensor([(20-epsilon)*1e-7, 400.0-epsilon, 10.0-epsilon, (15-epsilon)*1e-5, 1.0-epsilon, np.exp(5-epsilon)])
    ]
    
    for case_name, theta in zip(test_cases, test_values):
        print(f"\nTesting {case_name}:")
        try:
            log_prob = prior.log_prob(theta)
            print(f"  log_prob: {log_prob.item():.6f}")
            print(f"  finite:   {torch.isfinite(log_prob).item()}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    debug_edge_case()
    test_boundary_precision()