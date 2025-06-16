#!/usr/bin/env python3
"""
Script to generate training data for NPE on Lotka-Volterra model.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_generation import LVDataGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate LV training data for NPE')
    
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/training_data.pkl',
                       help='Output filepath')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Simulation parameters
    parser.add_argument('--x0_prey', type=int, default=50,
                       help='Initial prey population')
    parser.add_argument('--x0_pred', type=int, default=100,
                       help='Initial predator population')
    parser.add_argument('--t_max', type=float, default=10.0,
                       help='Maximum simulation time')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Time step for interpolation')
    
    parser.add_argument('--summary_stats', action='store_true',
                       help='Use summary statistics instead of full trajectory')
    
    args = parser.parse_args()
    
    print("Generating Lotka-Volterra training data...")
    print(f"Samples: {args.n_samples}")
    print(f"Initial conditions: ({args.x0_prey}, {args.x0_pred})")
    print(f"Time range: [0, {args.t_max}] with dt={args.dt}")
    print(f"Output: {args.output}")
    print(f"Using summary stats: {args.summary_stats}")
    print()
    
    # Initialize generator
    generator = LVDataGenerator(
        x0=(args.x0_prey, args.x0_pred),
        t_max=args.t_max,
        dt=args.dt,
        use_summary_stats=args.summary_stats,
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
        
        print(f"\nData generation completed successfully!")
        print(f"Generated {len(theta)} valid samples")
        print(f"Data shape: theta={theta.shape}, x={x.shape}")
        
        # Print some statistics
        stats = generator.get_stats()
        print(f"Success rate: {stats['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()