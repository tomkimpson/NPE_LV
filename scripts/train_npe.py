#!/usr/bin/env python3
"""
Script to train NPE on Lotka-Volterra data.
"""
import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import LVInference
from data_generation import LVDataGenerator


def main():
    parser = argparse.ArgumentParser(description='Train NPE on LV data')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--output', type=str, default='models/npe_model.pkl',
                       help='Output model filepath')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--validation_fraction', type=float, default=0.1,
                       help='Fraction of data for validation')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--hidden_features', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num_transforms', type=int, default=5,
                       help='Number of coupling transforms')
    
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("Training Neural Posterior Estimator...")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    # Load data
    try:
        theta, x, metadata = LVDataGenerator.load_data(args.data)
        print(f"Loaded {len(theta)} training samples")
        print(f"Parameter shape: {theta.shape}")
        print(f"Observation shape: {x.shape}")
        print()
        
        use_summary_stats = metadata.get('use_summary_stats', False)
        print(f"Using summary statistics: {use_summary_stats}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Initialize inference
    inference = LVInference(
        use_summary_stats=use_summary_stats,
        device=args.device,
        seed=args.seed
    )
    
    # Setup neural network
    neural_net_kwargs = {
        'hidden_features': args.hidden_features,
        'num_transforms': args.num_transforms,
    }
    
    inference.setup_inference(
        x_dim=x.shape[1],
        neural_net_kwargs=neural_net_kwargs
    )
    
    # Train
    try:
        training_info = inference.train(
            theta=theta,
            x=x,
            training_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_num_epochs=args.max_epochs,
            validation_fraction=args.validation_fraction,
            stop_after_epochs=args.early_stopping
        )
        
        print("Training completed successfully!")
        
        # Print training statistics
        try:
            if hasattr(training_info, '__contains__') and 'train_log_probs' in training_info:
                final_train_loss = training_info['train_log_probs'][-1]
                final_val_loss = training_info['validation_log_probs'][-1]
                print(f"Final training loss: {final_train_loss:.4f}")
                print(f"Final validation loss: {final_val_loss:.4f}")
            else:
                print("Training info format not recognized - skipping loss display")
        except Exception as e:
            print(f"Could not display training statistics: {e}")
        
        # Save model (exclude potentially problematic training_info)
        save_metadata = {
            'training_args': vars(args),
            'data_metadata': metadata,
        }
        
        inference.save_model(args.output, save_metadata)
        
        print(f"\nModel saved to {args.output}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()