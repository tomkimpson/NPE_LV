#!/usr/bin/env python3
"""
Script to run inference on observed data using trained NPE model.
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import LVInference
from data_generation import LVDataGenerator
from simulator import gillespie_lv
from utils import create_time_grid, flatten_trajectory, compute_summary_stats


def generate_synthetic_observation(theta_true, x0, t_max, dt, use_summary_stats=False, seed=None):
    """Generate synthetic observation for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    t_grid = create_time_grid(t_max, dt)
    
    # Simulate true trajectory
    _, trajectory = gillespie_lv(
        theta=theta_true,
        x0=x0,
        t_max=t_max,
        t_grid=t_grid
    )
    
    # Convert to observation format
    if use_summary_stats:
        x_obs = compute_summary_stats(trajectory, t_grid)
    else:
        x_obs = flatten_trajectory(trajectory)
    
    return torch.tensor(x_obs, dtype=torch.float32), trajectory


def main():
    parser = argparse.ArgumentParser(description='Run inference using trained NPE')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained NPE model')
    parser.add_argument('--observation', type=str, default=None,
                       help='Path to observation data (optional)')
    
    # Synthetic observation parameters (if no real data provided)
    parser.add_argument('--theta_true', type=float, nargs=4, 
                       default=[0.5, 0.025, 0.025, 0.5],
                       help='True parameters for synthetic observation [α β δ γ]')
    parser.add_argument('--x0_prey', type=int, default=50,
                       help='Initial prey population')
    parser.add_argument('--x0_pred', type=int, default=100,
                       help='Initial predator population')
    parser.add_argument('--t_max', type=float, default=10.0,
                       help='Maximum time for synthetic observation')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Time step for synthetic observation')
    
    # Inference parameters
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of posterior samples')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("Running NPE inference...")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    try:
        inference = LVInference.load_model(args.model)
        print("Loaded trained NPE model")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load or generate observation
    if args.observation is not None:
        # Load real observation data
        try:
            # Assume observation is saved as torch tensor
            x_obs = torch.load(args.observation)
            print(f"Loaded observation from {args.observation}")
            theta_true = None
            trajectory_true = None
            
        except Exception as e:
            print(f"Error loading observation: {e}")
            sys.exit(1)
    else:
        # Generate synthetic observation
        print("Generating synthetic observation...")
        theta_true = np.array(args.theta_true)
        print(f"True parameters: {theta_true}")
        
        x_obs, trajectory_true = generate_synthetic_observation(
            theta_true=theta_true,
            x0=(args.x0_prey, args.x0_pred),
            t_max=args.t_max,
            dt=args.dt,
            use_summary_stats=inference.use_summary_stats,
            seed=args.seed
        )
        
        print(f"Generated observation with shape: {x_obs.shape}")
    
    # Sample from posterior
    print(f"\nSampling {args.num_samples} samples from posterior...")
    try:
        posterior_samples = inference.sample_posterior(
            x_obs=x_obs,
            num_samples=args.num_samples
        )
        
        print("Posterior sampling completed!")
        
        # Compute summary statistics
        samples_np = posterior_samples.numpy()
        param_names = ['α', 'β', 'δ', 'γ']
        
        print("\nPosterior summary:")
        print("-" * 50)
        for i, name in enumerate(param_names):
            mean_val = np.mean(samples_np[:, i])
            std_val = np.std(samples_np[:, i])
            q025 = np.percentile(samples_np[:, i], 2.5)
            q975 = np.percentile(samples_np[:, i], 97.5)
            
            print(f"{name}: {mean_val:.4f} ± {std_val:.4f} "
                  f"[{q025:.4f}, {q975:.4f}]", end="")
            
            if theta_true is not None:
                print(f" (true: {theta_true[i]:.4f})")
            else:
                print()
        
        # Save results
        results = {
            'posterior_samples': posterior_samples,
            'observation': x_obs,
            'theta_true': torch.tensor(theta_true) if theta_true is not None else None,
            'args': vars(args)
        }
        
        results_path = Path(args.output_dir) / 'inference_results.pkl'
        torch.save(results, results_path)
        print(f"\nResults saved to {results_path}")
        
        # Create plots
        print("Creating plots...")
        
        # Marginal posteriors
        fig1 = inference.plot_posterior_samples(
            posterior_samples,
            true_theta=torch.tensor(theta_true) if theta_true is not None else None
        )
        fig1.savefig(Path(args.output_dir) / 'posterior_marginals.png', dpi=150, bbox_inches='tight')
        print("Saved posterior marginals plot")
        
        # Pairwise plot
        fig2 = inference.plot_pairwise(
            posterior_samples,
            true_theta=torch.tensor(theta_true) if theta_true is not None else None
        )
        fig2.savefig(Path(args.output_dir) / 'posterior_pairwise.png', dpi=150, bbox_inches='tight')
        print("Saved pairwise posterior plot")
        
        # True trajectory plot (if available)
        if trajectory_true is not None:
            fig3, ax = plt.subplots(figsize=(10, 6))
            t_grid = create_time_grid(args.t_max, args.dt)
            
            ax.plot(t_grid, trajectory_true[:, 0], 'b-', label='Prey (true)', linewidth=2)
            ax.plot(t_grid, trajectory_true[:, 1], 'r-', label='Predator (true)', linewidth=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.set_title('True Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig3.savefig(Path(args.output_dir) / 'true_trajectory.png', dpi=150, bbox_inches='tight')
            print("Saved true trajectory plot")
        
        plt.close('all')
        
        print(f"\nInference completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()