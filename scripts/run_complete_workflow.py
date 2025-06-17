#!/usr/bin/env python3
"""
Complete end-to-end workflow script for NPE on Lotka-Volterra model.
Generates data, trains model, and runs inference in a single script.
"""
import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {description} completed in {elapsed:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED: {description} failed after {elapsed:.1f} seconds")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Complete NPE-LV workflow')
    
    # Data generation parameters
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for data generation')
    parser.add_argument('--t_max', type=float, default=10.0,
                       help='Maximum simulation time')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='Time step for interpolation')
    parser.add_argument('--summary_stats', action='store_true',
                       help='Use summary statistics instead of full trajectories')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--train_batch_size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_features', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num_transforms', type=int, default=5,
                       help='Number of coupling transforms')
    
    # Inference parameters
    parser.add_argument('--theta_true', type=float, nargs=4, 
                       default=[0.5, 0.025, 0.025, 0.5],
                       help='True parameters for inference test [α β δ γ]')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of posterior samples')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--workflow_name', type=str, default=None,
                       help='Name for this workflow run (default: timestamped)')
    
    # Skip steps (for partial runs)
    parser.add_argument('--skip_data', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training (use existing model)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to existing data (if skipping data generation)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to existing model (if skipping training)')
    
    args = parser.parse_args()
    
    # Create workflow directory
    if args.workflow_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_name = f"workflow_{timestamp}"
    else:
        workflow_name = args.workflow_name
    
    workflow_dir = Path(f"workflows/{workflow_name}")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    data_path = args.data_path or str(workflow_dir / "training_data.pkl")
    model_path = args.model_path or str(workflow_dir / "npe_model.pkl") 
    results_dir = str(workflow_dir / "results")
    
    print(f"🚀 Starting NPE-LV Complete Workflow: {workflow_name}")
    print(f"📁 Workflow directory: {workflow_dir}")
    print(f"🎯 Target parameters: {args.theta_true}")
    print(f"📊 Training samples: {args.n_samples}")
    print(f"🔧 Device: {args.device}")
    
    # Save workflow configuration
    config_path = workflow_dir / "config.txt"
    with open(config_path, 'w') as f:
        f.write(f"NPE-LV Workflow Configuration\\n")
        f.write(f"{'='*40}\\n")
        f.write(f"Timestamp: {datetime.now()}\\n")
        f.write(f"Workflow name: {workflow_name}\\n\\n")
        
        f.write("Parameters:\\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\\n")
    
    total_start = time.time()
    
    # Step 1: Data Generation
    if not args.skip_data:
        data_cmd = [
            'python', 'scripts/generate_data.py',
            '--n_samples', str(args.n_samples),
            '--output', data_path,
            '--batch_size', str(args.batch_size),
            '--t_max', str(args.t_max),
            '--dt', str(args.dt),
            '--seed', str(args.seed)
        ]
        
        if args.summary_stats:
            data_cmd.append('--summary_stats')
            
        if not run_command(data_cmd, "Data Generation"):
            print("❌ Workflow failed at data generation step")
            sys.exit(1)
    else:
        print(f"⏭️  Skipping data generation, using: {data_path}")
    
    # Step 2: Model Training  
    if not args.skip_training:
        train_cmd = [
            'python', 'scripts/train_npe.py',
            '--data', data_path,
            '--output', model_path,
            '--max_epochs', str(args.max_epochs),
            '--batch_size', str(args.train_batch_size),
            '--learning_rate', str(args.learning_rate),
            '--hidden_features', str(args.hidden_features),
            '--num_transforms', str(args.num_transforms),
            '--device', args.device,
            '--seed', str(args.seed)
        ]
        
        if not run_command(train_cmd, "Model Training"):
            print("❌ Workflow failed at training step")
            sys.exit(1)
    else:
        print(f"⏭️  Skipping training, using: {model_path}")
    
    # Step 3: Inference
    infer_cmd = [
        'python', 'scripts/run_inference.py',
        '--model', model_path,
        '--theta_true'] + [str(x) for x in args.theta_true] + [
        '--num_samples', str(args.num_samples),
        '--output_dir', results_dir,
        '--t_max', str(args.t_max),
        '--dt', str(args.dt),
        '--seed', str(args.seed + 1000)  # Different seed for inference
    ]
    
    if not run_command(infer_cmd, "Inference"):
        print("❌ Workflow failed at inference step")
        sys.exit(1)
    
    # Workflow completed
    total_elapsed = time.time() - total_start
    
    print(f"\\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"📁 All results saved in: {workflow_dir}")
    print(f"\\n📊 Results summary:")
    print(f"   📈 Training data: {data_path}")
    print(f"   🧠 Trained model: {model_path}")
    print(f"   📊 Inference results: {results_dir}")
    print(f"   ⚙️  Configuration: {config_path}")
    
    print(f"\\n🔍 Next steps:")
    print(f"   - Check posterior plots in {results_dir}/")
    print(f"   - Open visualization notebook for detailed analysis")
    print(f"   - Try different parameter values for inference")


if __name__ == '__main__':
    main()