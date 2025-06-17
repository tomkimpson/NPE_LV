#!/usr/bin/env python3
"""
Script to generate training data for TEIRV NPE.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_data_generation import TEIRVDataGenerator


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
                       help='Time step for observations (days)')
    parser.add_argument('--observation_noise', type=float, default=1.0,
                       help='RT-PCR observation noise (log10 scale)')
    parser.add_argument('--detection_limit', type=float, default=-0.65,
                       help='RT-PCR detection limit (log10 scale)')
    
    parser.add_argument('--full_trajectory', action='store_true',
                       help='Use full trajectory instead of RT-PCR observations only')
    
    args = parser.parse_args()
    
    print("Generating TEIRV training data...")
    print(f"Samples: {args.n_samples}")
    print(f"Time range: [0, {args.t_max}] days with dt={args.dt}")
    print(f"Observation type: {'Full trajectory' if args.full_trajectory else 'RT-PCR only'}")
    print(f"RT-PCR noise: {args.observation_noise}")
    print(f"Detection limit: {args.detection_limit}")
    print(f"Output: {args.output}")
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
    
    # Generate data
    try:
        theta, x = generator.generate_batch(
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
        
        # Save data
        metadata = {
            'script_args': vars(args),
            'generation_stats': generator.get_stats()
        }
        
        generator.save_data(theta, x, args.output, metadata)
        
        print(f"\\nData generation completed successfully!")
        print(f"Generated {len(theta)} valid samples")
        print(f"Data shapes: theta={theta.shape}, x={x.shape}")
        
        # Print generation statistics
        stats = generator.get_stats()
        print(f"\\nGeneration Statistics:")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Observation type: {stats['observation_type']}")
        print(f"  Time points: {stats['time_points']}")
        print(f"  RT-PCR noise: {stats['observation_noise']}")
        print(f"  Detection limit: {stats['detection_limit']}")
        
        # Show parameter summary
        print(f"\\nParameter Summary:")
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        theta_np = theta.numpy()
        for i, name in enumerate(param_names):
            mean_val = theta_np[:, i].mean()
            std_val = theta_np[:, i].std()
            min_val = theta_np[:, i].min()
            max_val = theta_np[:, i].max()
            print(f"  {name}: {mean_val:.2f} ± {std_val:.2f} [{min_val:.2f}, {max_val:.2f}]")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()