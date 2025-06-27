#!/usr/bin/env python3
"""
Script to generate training data for TEIRV NPE.
"""
import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_data_generation import TEIRVDataGenerator
from TEIRV.teirv_utils import create_teirv_prior


def main():
    parser = argparse.ArgumentParser(description='Generate TEIRV training data for NPE')
    
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/teirv_training_data.pkl',
                       help='Output filepath')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Simulation parameters
    parser.add_argument('--t_max', type=float, default=14.0,
                       help='Maximum simulation time (days)')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Observation grid interval - for interpolating Gillespie results (days)')
    parser.add_argument('--observation_noise', type=float, default=1.0,
                       help='RT-PCR observation noise (log10 scale)')
    parser.add_argument('--detection_limit', type=float, default=-0.65,
                       help='RT-PCR detection limit (log10 scale)')
    
    parser.add_argument('--full_trajectory', action='store_true',
                       help='Use full trajectory instead of RT-PCR observations only')
    
    args = parser.parse_args()
    
    print("TEIRV Training Data Generation")
    print("=" * 50)
    print(f"Samples: {args.n_samples}")
    print(f"Simulation timespan: [0, {args.t_max}] days")
    n_timepoints = int(args.t_max / args.dt) + 1
    print(f"Observation grid: {n_timepoints} points at {args.dt}-day intervals")
    print(f"Observation type: {'Full trajectory' if args.full_trajectory else 'RT-PCR only'}")
    print(f"RT-PCR noise: {args.observation_noise}")
    print(f"Detection limit: {args.detection_limit}")
    print(f"Output: {args.output}")
    print()
    
    # Display prior distributions
    print("TEIRV Parameter Priors:")
    print("-" * 30)
    prior = create_teirv_prior()
    print(f"  β (infection rate):       Uniform({prior.beta_bounds[0]}, {prior.beta_bounds[1]})")
    print(f"  π (virion production):    Uniform({prior.pi_bounds[0]}, {prior.pi_bounds[1]})")
    print(f"  δ (cell clearance):       Uniform({prior.delta_bounds[0]}, {prior.delta_bounds[1]})")
    print(f"  φ (interferon protection): Uniform({prior.phi_bounds[0]}, {prior.phi_bounds[1]})")
    print(f"  ρ (reversion rate):       Uniform({prior.rho_bounds[0]}, {prior.rho_bounds[1]})")
    print(f"  V₀ (initial virions):     exp(Uniform({prior.lnv0_bounds[0]}, {prior.lnv0_bounds[1]})) ≈ [{int(np.exp(prior.lnv0_bounds[0]))}, {int(np.exp(prior.lnv0_bounds[1]))}]")
    print()
    
    # Initialize generator
    generator = TEIRVDataGenerator(
        t_max=args.t_max,
        dt=args.dt,
        observation_noise=args.observation_noise,
        detection_limit=args.detection_limit,
        use_observations_only=not args.full_trajectory,
        seed=args.seed
    )
    
    # Generate data with timing
    print("Starting data generation...")
    start_time = time.time()
    
    try:
        theta, x = generator.generate_batch(
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
        
        generation_time = time.time() - start_time
        
        # Save data
        metadata = {
            'script_args': vars(args),
            'generation_stats': generator.get_stats()
        }
        
        generator.save_data(theta, x, args.output, metadata)
        
        print(f"\n" + "="*50)
        print(f"DATA GENERATION COMPLETED SUCCESSFULLY!")
        print(f"="*50)
        print(f"⏱️  Generation time: {generation_time:.1f} seconds")
        print(f"📊 Generated {len(theta)} valid samples")
        print(f"📏 Data shapes: theta={theta.shape}, x={x.shape}")
        print(f"⚡ Generation rate: {len(theta)/generation_time:.1f} samples/second")
        
        # Print generation statistics
        stats = generator.get_stats()
        print(f"\nGeneration Statistics:")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Observation type: {stats['observation_type']}")
        print(f"  Time points: {stats['time_points']}")
        print(f"  RT-PCR noise: {stats['observation_noise']}")
        print(f"  Detection limit: {stats['detection_limit']}")
        
        # Show parameter summary
        print(f"\nParameter Summary (Generated Samples):")
        print("-" * 40)
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        theta_np = theta.numpy()
        for i, name in enumerate(param_names):
            mean_val = theta_np[:, i].mean()
            std_val = theta_np[:, i].std()
            min_val = theta_np[:, i].min()
            max_val = theta_np[:, i].max()
            print(f"  {name}: {mean_val:.2f} ± {std_val:.2f} [{min_val:.2f}, {max_val:.2f}]")
        
        # Time coverage summary
        print(f"\nTemporal Coverage:")
        print(f"  Simulation timespan: {args.t_max} days per sample")
        print(f"  Observation grid: {args.dt}-day intervals (interpolated from Gillespie)")
        print(f"  Data points per sample: {n_timepoints}")
        print(f"  Total dataset coverage: {len(theta) * args.t_max:.0f} patient-days")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()