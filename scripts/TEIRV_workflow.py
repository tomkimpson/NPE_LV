#!/usr/bin/env python3
"""
Complete TEIRV NPE Workflow Script

This script provides a unified interface for all TEIRV Neural Posterior Estimation tasks:
- Data generation for training
- NPE model training
- Clinical inference on real patient data
- Validation against particle filter benchmarks
- Complete end-to-end pipeline

Usage:
    python TEIRV_workflow.py generate --n_samples 50000
    python TEIRV_workflow.py train --data data/training.pkl
    python TEIRV_workflow.py inference --model models/npe.pkl
    python TEIRV_workflow.py demo
    python TEIRV_workflow.py full --n_samples 10000
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_data_generation import TEIRVDataGenerator
from TEIRV.teirv_inference import TEIRVInference
from TEIRV.clinical_data import ClinicalStudy, validate_clinical_data_compatibility
from TEIRV.teirv_utils import create_teirv_prior, create_teirv_time_grid, teirv_parameter_summary, get_teirv_initial_conditions, apply_observation_model
from TEIRV.teirv_simulator import gillespie_teirv


class TEIRVWorkflow:
    """Main workflow class for TEIRV NPE operations."""
    
    def __init__(self, seed: int = 42, device: str = 'cpu'):
        self.seed = seed
        self.device = device
        self.workflow_dir = None
        
        # Set consistent float32 precision throughout
        torch.set_default_dtype(torch.float32)
        
    def setup_workflow_directory(self, workflow_name: Optional[str] = None) -> Path:
        """Set up directory for workflow outputs."""
        if workflow_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workflow_name = f"teirv_workflow_{timestamp}"
        
        self.workflow_dir = Path(f"workflows/{workflow_name}")
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        return self.workflow_dir
    
    def generate_data(self, args) -> str:
        """Generate training data for TEIRV NPE."""
        print("🔬 TEIRV Training Data Generation")
        print("=" * 50)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if data already exists
        if output_path.exists():
            print(f"✅ Training data already exists: {output_path}")
            print(f"⏭️  Skipping data generation...")
            return str(output_path)
        
        # Display configuration
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
        
        # Generate data
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
            
            print(f"\n✅ DATA GENERATION COMPLETED")
            print(f"⏱️  Generation time: {generation_time:.1f} seconds")
            print(f"📊 Generated {len(theta)} valid samples")
            print(f"⚡ Generation rate: {len(theta)/generation_time:.1f} samples/second")
            
            return args.output
            
        except Exception as e:
            print(f"❌ Error during data generation: {e}")
            sys.exit(1)
    
    def train_model(self, args) -> str:
        """Train NPE model on TEIRV data."""
        print("🧠 TEIRV NPE Model Training")
        print("=" * 50)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Data: {args.data}")
        print(f"Output: {args.output}")
        print(f"Device: {args.device}")
        print()
        
        # Load data
        try:
            theta, x, metadata = TEIRVDataGenerator.load_data(args.data)
            print(f"✅ Loaded {len(theta)} training samples")
            print(f"Parameter shape: {theta.shape}")
            print(f"Observation shape: {x.shape}")
            
            observation_type = 'rt_pcr' if metadata.get('use_observations_only', True) else 'full_trajectory'
            print(f"Observation type: {observation_type}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
        
        # Initialize inference
        inference = TEIRVInference(
            observation_type=observation_type,
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
        print("Starting training...")
        start_time = time.time()
        
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
            
            training_time = time.time() - start_time
            
            # Save model
            save_metadata = {
                'training_args': vars(args),
                'data_metadata': metadata,
                'model_type': 'TEIRV_NPE',
                'observation_type': observation_type
            }
            
            inference.save_model(args.output, save_metadata)
            
            print(f"\n✅ TRAINING COMPLETED")
            print(f"⏱️  Training time: {training_time:.1f} seconds")
            print(f"📁 Model saved to {args.output}")
            
            return args.output
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            sys.exit(1)
    
    def inference(self, args) -> str:
        """Perform inference on patient data."""
        print("🏥 TEIRV Inference")
        print("=" * 50)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load NPE model
        print(f"Loading NPE model from {args.model}")
        try:
            npe_model = TEIRVInference(device=args.device)
            npe_model.load_model(args.model)
            print("✅ NPE model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load NPE model: {e}")
            sys.exit(1)
        
        # Load clinical data
        print("\nLoading clinical data...")
        try:
            study = ClinicalStudy(args.data_dir)
            summary_df = study.get_study_summary()
            print(f"✅ Found {len(study.loader.patient_ids)} patients")
            
            # Filter patients
            if args.patients is None:
                good_patients = study.filter_patients(
                    min_detections=args.min_detections,
                    min_peak_viral_load=args.min_peak_vl
                )
            else:
                good_patients = args.patients
            
            print(f"✅ Processing {len(good_patients)} patients")
            
        except Exception as e:
            print(f"❌ Error loading clinical data: {e}")
            sys.exit(1)
        
        # Prepare data for inference
        time_grid = create_teirv_time_grid(14.0, 1.0)
        patient_data = study.prepare_for_inference(good_patients, time_grid)
        
        # Perform inference
        print(f"\nPerforming NPE inference ({args.n_samples} samples per patient)...")
        results = {}
        total_start = time.time()
        
        for patient_id in good_patients:
            if patient_id not in patient_data:
                print(f"⚠️  Skipping patient {patient_id}")
                continue
            
            print(f"\nProcessing patient {patient_id}")
            observations = patient_data[patient_id]
            
            try:
                start_time = time.time()
                posterior_samples = npe_model.sample_posterior(
                    observations.unsqueeze(0),
                    num_samples=args.n_samples
                )
                inference_time = time.time() - start_time
                
                # Move tensors to CPU for numpy operations
                posterior_cpu = posterior_samples.squeeze(0).cpu()
                observations_cpu = observations.cpu()
                
                results[patient_id] = {
                    'posterior_samples': posterior_cpu,
                    'observations': observations_cpu,
                    'parameter_summary': teirv_parameter_summary(posterior_cpu),
                    'inference_time': inference_time
                }
                
                print(f"  ✅ Completed in {inference_time:.2f}s")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        total_time = time.time() - total_start
        
        # Save results
        self._save_clinical_results(results, output_path, args)
        
        print(f"\n✅ CLINICAL INFERENCE COMPLETED")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Results saved to {output_path}")
        
        return str(output_path)
    
    def _save_clinical_results(self, results: Dict, output_path: Path, args):
        """Save clinical inference results."""
        # Create summary DataFrame
        rows = []
        for patient_id, result in results.items():
            summary = result['parameter_summary']
            row = {'patient_id': patient_id}
            
            for param, stats in summary.items():
                row[f'{param}_mean'] = stats['mean']
                row[f'{param}_q025'] = stats['q025']
                row[f'{param}_q975'] = stats['q975']
            
            row['inference_time'] = result['inference_time']
            row['n_samples'] = len(result['posterior_samples'])
            rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_path / 'clinical_parameter_estimates.csv', index=False)
        
        # Save individual patient results and create plots
        for patient_id, result in results.items():
            patient_dir = output_path / f'patient_{patient_id}'
            patient_dir.mkdir(exist_ok=True)
            
            # Save samples and observations
            np.save(patient_dir / 'posterior_samples.npy', result['posterior_samples'].numpy())
            np.save(patient_dir / 'observations.npy', result['observations'].numpy())
            
            # Save parameter summary
            summary = result['parameter_summary']
            pd.DataFrame(summary).T.to_csv(patient_dir / 'parameter_summary.csv')
            
            # Create corner plot for this patient
            print(f"Creating corner plot for patient {patient_id}")
            try:
                # Initialize a temporary inference object for plotting
                temp_inference = TEIRVInference()
                fig_corner = temp_inference.plot_corner(result['posterior_samples'])
                fig_corner.savefig(patient_dir / f'patient_{patient_id}_corner.png', 
                                 dpi=300, bbox_inches='tight')
                plt.close(fig_corner)
                print(f"  ✅ Corner plot saved")
            except Exception as e:
                print(f"  ❌ Corner plot failed: {e}")
            
            # Create predictive plot with credible intervals
            print(f"Creating predictive plot for patient {patient_id}")
            try:
                fig_pred = self._create_predictive_plot(
                    patient_id, result['posterior_samples'], result['observations']
                )
                fig_pred.savefig(patient_dir / f'patient_{patient_id}_predictive.png',
                               dpi=300, bbox_inches='tight')
                plt.close(fig_pred)
                print(f"  ✅ Predictive plot saved")
            except Exception as e:
                print(f"  ❌ Predictive plot failed: {e}")
        
        # Create summary plots
        self._create_clinical_plots(results, output_path)
    
    def _create_predictive_plot(self, patient_id: str, posterior_samples: torch.Tensor, 
                               observations: torch.Tensor, n_pred_samples: int = 20) -> plt.Figure:
        """
        Create predictive plot with credible intervals.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        posterior_samples : torch.Tensor
            Posterior parameter samples
        observations : torch.Tensor
            Observed RT-PCR data
        n_pred_samples : int
            Number of posterior samples to use for predictions
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Predictive plot figure
        """
        print(f"  Generating {n_pred_samples} posterior predictions...")
        
        # Timing for progress estimation
        import time
        start_time = time.time()
        first_sim_time = None
        
        # Time grids
        t_obs = np.arange(0, 15, 1.0)  # Observed range: 0-14 days
        t_pred = np.arange(0, 21, 1.0)  # Extended range: 0-20 days
        
        # Select subset of posterior samples
        n_samples = min(n_pred_samples, len(posterior_samples))
        sample_indices = np.random.choice(len(posterior_samples), n_samples, replace=False)
        selected_samples = posterior_samples[sample_indices]
        
        # Generate predictions
        predictions_obs = []  # For observed range
        predictions_ext = []  # For extended range
        
        base_ic = get_teirv_initial_conditions()
        
        for i, theta_sample in enumerate(selected_samples):
            sim_start = time.time()
            
            # Progress reporting
            if i % 5 == 0 or i == 0:
                print(f"    Processing sample {i+1}/{n_samples}...")
            try:
                # Set up initial conditions
                ic = base_ic.copy()
                ic['V'] = theta_sample[5].item()  # V₀ from posterior
                
                # Simulate for extended range
                _, trajectory_ext = gillespie_teirv(
                    theta=theta_sample.numpy(),
                    initial_conditions=ic,
                    t_max=20.0,
                    t_grid=t_pred,
                    max_steps=1000000
                )
                
                # Apply observation model (RT-PCR transformation)
                V_trajectory_ext = trajectory_ext[:, 4]  # V compartment
                obs_ext = apply_observation_model(
                    V_trajectory=V_trajectory_ext,
                    sigma=1.0,  # Standard observation noise
                    detection_limit=-0.65,
                    add_noise=True
                )
                
                # Extract observed range (first 15 points: 0-14 days)
                obs_range = obs_ext[:15]
                
                predictions_obs.append(obs_range)
                predictions_ext.append(obs_ext)
                
                # Time estimation after first simulation
                if i == 0:
                    first_sim_time = time.time() - sim_start
                    estimated_total = first_sim_time * n_samples
                    print(f"    Estimated total simulation time: {estimated_total:.1f}s ({estimated_total/60:.1f} min)")
                
            except Exception as e:
                print(f"    Simulation {i} failed: {e}")
                continue
        
        if len(predictions_obs) == 0:
            print("  ❌ No successful predictions generated")
            return plt.figure()
        
        # Convert to arrays and compute credible intervals
        pred_obs_array = np.array(predictions_obs)  # Shape: (n_samples, 15)
        pred_ext_array = np.array(predictions_ext)  # Shape: (n_samples, 21)
        
        # Compute quantiles
        quantiles = [0.025, 0.125, 0.25, 0.5, 0.75, 0.875, 0.975]  # 0%, 25%, 50%, 75%, 95%
        
        # Observed range credible intervals
        ci_obs = np.percentile(pred_obs_array, [2.5, 12.5, 25, 50, 75, 87.5, 97.5], axis=0)
        
        # Extended range credible intervals  
        ci_ext = np.percentile(pred_ext_array, [2.5, 12.5, 25, 50, 75, 87.5, 97.5], axis=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot observed data points
        ax.scatter(t_obs, observations.numpy(), color='black', s=80, 
                  label='Observed data', zorder=10, alpha=0.8)
        
        # Plot credible intervals for observed range (red)
        # 95% CI
        ax.fill_between(t_obs, ci_obs[0], ci_obs[6], alpha=0.2, color='red', 
                       label='95% CI (observed)')
        # 75% CI
        ax.fill_between(t_obs, ci_obs[1], ci_obs[5], alpha=0.3, color='red')
        # 50% CI  
        ax.fill_between(t_obs, ci_obs[2], ci_obs[4], alpha=0.4, color='red')
        # Median (observed range)
        ax.plot(t_obs, ci_obs[3], color='darkred', linewidth=2, label='Median (observed)')
        
        # Plot credible intervals for extended range (purple) - connected to observed range
        t_pred_ext = t_pred[14:]  # Start from day 14 to connect with observed range
        ci_ext_pred = ci_ext[:, 14:]  # Start from day 14 to connect
        
        # 95% CI
        ax.fill_between(t_pred_ext, ci_ext_pred[0], ci_ext_pred[6], alpha=0.2, color='purple',
                       label='95% CI (predicted)')
        # 75% CI
        ax.fill_between(t_pred_ext, ci_ext_pred[1], ci_ext_pred[5], alpha=0.3, color='purple')
        # 50% CI
        ax.fill_between(t_pred_ext, ci_ext_pred[2], ci_ext_pred[4], alpha=0.4, color='purple')
        # Median (predicted range) - connected to observed range
        ax.plot(t_pred_ext, ci_ext_pred[3], color='darkviolet', linewidth=2, 
               label='Median (predicted)')
        
        # Add vertical line at transition
        ax.axvline(x=14, color='gray', linestyle='--', alpha=0.7, 
                  label='Observed/Predicted boundary')
        
        # Formatting
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('log₁₀ Viral Load', fontsize=12)
        ax.set_title(f'Patient {patient_id}: Posterior Predictive Check', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set reasonable y-limits for log10 viral load data
        all_data = np.concatenate([observations.numpy(), ci_obs.flatten(), ci_ext.flatten()])
        y_min = np.min(all_data) - 0.5  # Add some padding below
        y_max = np.max(all_data) + 0.5  # Add some padding above
        ax.set_ylim(y_min, y_max)
        
        # Add horizontal line for detection limit
        ax.axhline(y=-0.65, color='gray', linestyle=':', alpha=0.7, 
                  label='Detection limit')
        
        plt.tight_layout()
        
        print(f"  ✅ Generated predictions from {len(predictions_obs)} successful simulations")
        return fig
    
    def _create_clinical_plots(self, results: Dict, output_path: Path):
        """Create clinical summary plots."""
        param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
        
        # Parameter estimates plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(param_names):
            ax = axes[i]
            
            means = []
            lowers = []
            uppers = []
            patients = []
            
            for patient_id, result in results.items():
                summary = result['parameter_summary']
                if param in summary:
                    means.append(summary[param]['mean'])
                    lowers.append(summary[param]['q025'])
                    uppers.append(summary[param]['q975'])
                    patients.append(patient_id[:6])
            
            if means:
                x_pos = range(len(means))
                ax.errorbar(x_pos, means, 
                           yerr=[np.array(means) - np.array(lowers), 
                                 np.array(uppers) - np.array(means)],
                           fmt='o', capsize=5)
                ax.set_xlabel('Patient')
                ax.set_ylabel(param)
                ax.set_title(f'Parameter {param}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(patients, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_estimates_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def demo(self, args):
        """Run quick demo of TEIRV NPE."""
        print("🎬 TEIRV NPE Demo")
        print("=" * 30)
        
        # Setup demo directory
        demo_dir = self.setup_workflow_directory("demo")
        
        print("This demo will:")
        print("1. Generate small training dataset (1000 samples)")
        print("2. Train a basic NPE model")
        print("3. Run inference on one clinical patient")
        print()
        
        # Generate demo data
        print("📊 Generating demo training data...")
        demo_args = argparse.Namespace(
            n_samples=1000,
            output=str(demo_dir / "demo_data.pkl"),
            batch_size=500,
            seed=42,
            t_max=14.0,
            dt=1.0,
            observation_noise=1.0,
            detection_limit=-0.65,
            full_trajectory=False
        )
        
        data_path = self.generate_data(demo_args)
        
        # Train demo model
        print("\n🧠 Training demo NPE model...")
        train_args = argparse.Namespace(
            data=data_path,
            output=str(demo_dir / "demo_model.pkl"),
            batch_size=256,
            learning_rate=5e-4,
            max_epochs=50,
            validation_fraction=0.2,
            early_stopping=10,
            hidden_features=128,
            num_transforms=4,
            device=self.device,
            seed=42
        )
        
        model_path = self.train_model(train_args)
        
        # Run inference
        print("\n🏥 Running inference...")
        inference_args = argparse.Namespace(
            model=model_path,
            output=str(demo_dir / "inference_results"),
            n_samples=5000,
            patients=None,
            min_detections=5,
            min_peak_vl=2.0,
            data_dir=None
        )
        
        self.inference(inference_args)
        
        print(f"\n✅ DEMO COMPLETED")
        print(f"📁 All results in: {demo_dir}")
    
    def full_pipeline(self, args):
        """Run complete end-to-end pipeline."""
        print("🚀 TEIRV NPE Complete Pipeline")
        print("=" * 50)
        
        workflow_dir = self.setup_workflow_directory(args.workflow_name)
        
        print(f"📁 Workflow directory: {workflow_dir}")
        print(f"📊 Training samples: {args.n_samples}")
        print(f"🔧 Device: {args.device}")
        print()
        
        total_start = time.time()
        
        # Step 1: Data Generation
        print("🔬 Step 1: Generating training data...")
        data_args = argparse.Namespace(
            n_samples=args.n_samples,
            output=str(workflow_dir / "training_data.pkl"),
            batch_size=args.batch_size,
            seed=args.seed,
            t_max=args.t_max,
            dt=args.dt,
            observation_noise=args.observation_noise,
            detection_limit=args.detection_limit,
            full_trajectory=args.full_trajectory
        )
        
        data_path = self.generate_data(data_args)
        
        # Step 2: Model Training
        print("\n🧠 Step 2: Training NPE model...")
        train_args = argparse.Namespace(
            data=data_path,
            output=str(workflow_dir / "npe_model.pkl"),
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            validation_fraction=args.validation_fraction,
            early_stopping=args.early_stopping,
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
            device=args.device,
            seed=args.seed
        )
        
        model_path = self.train_model(train_args)
        
        # Step 3: Inference
        print("\n🏥 Step 3: Running inference...")
        inference_args = argparse.Namespace(
            model=model_path,
            output=str(workflow_dir / "inference_results"),
            n_samples=args.inference_samples,
            patients=None,
            min_detections=args.min_detections,
            min_peak_vl=args.min_peak_vl,
            data_dir=None
        )
        
        self.inference(inference_args)
        
        total_time = time.time() - total_start
        
        print(f"\n🎉 COMPLETE PIPELINE FINISHED")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📁 All results in: {workflow_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='TEIRV NPE Complete Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data
  python TEIRV_workflow.py generate --n_samples 50000

  # Train NPE model
  python TEIRV_workflow.py train --data data/training.pkl

  # Run inference
  python TEIRV_workflow.py inference --model models/npe.pkl

  # Quick demo
  python TEIRV_workflow.py demo

  # Complete pipeline
  python TEIRV_workflow.py full --n_samples 20000
        """
    )
    
    # Global parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for computation')
    
    subparsers = parser.add_subparsers(dest='mode', help='Workflow mode')
    
    # Generate mode
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples')
    gen_parser.add_argument('--output', type=str, default='data/teirv_training_data.pkl', help='Output filepath')
    gen_parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    gen_parser.add_argument('--t_max', type=float, default=14.0, help='Max simulation time (days)')
    gen_parser.add_argument('--dt', type=float, default=1.0, help='Time step (days)')
    gen_parser.add_argument('--observation_noise', type=float, default=1.0, help='RT-PCR noise')
    gen_parser.add_argument('--detection_limit', type=float, default=-0.65, help='Detection limit')
    gen_parser.add_argument('--full_trajectory', action='store_true', help='Use full trajectory')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train NPE model')
    train_parser.add_argument('--data', type=str, required=True, help='Training data path')
    train_parser.add_argument('--output', type=str, default='models/teirv_npe_model.pkl', help='Output model path')
    train_parser.add_argument('--batch_size', type=int, default=512, help='Training batch size')
    train_parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    train_parser.add_argument('--max_epochs', type=int, default=150, help='Max epochs')
    train_parser.add_argument('--validation_fraction', type=float, default=0.15, help='Validation fraction')
    train_parser.add_argument('--early_stopping', type=int, default=25, help='Early stopping patience')
    train_parser.add_argument('--hidden_features', type=int, default=256, help='Hidden layer size')
    train_parser.add_argument('--num_transforms', type=int, default=8, help='Number of transforms')
    
    # Inference mode
    inference_parser = subparsers.add_parser('inference', help='Run inference on patient data')
    inference_parser.add_argument('--model', type=str, required=True, help='NPE model path')
    inference_parser.add_argument('--output', type=str, default='results/inference', help='Output directory')
    inference_parser.add_argument('--n_samples', type=int, default=10000, help='Posterior samples')
    inference_parser.add_argument('--patients', type=str, nargs='+', default=None, help='Specific patients')
    inference_parser.add_argument('--min_detections', type=int, default=5, help='Min detections')
    inference_parser.add_argument('--min_peak_vl', type=float, default=2.0, help='Min peak viral load')
    inference_parser.add_argument('--data_dir', type=str, default=None, help='Clinical data directory')
    
    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Quick demo')
    
    # Full pipeline mode
    full_parser = subparsers.add_parser('full', help='Complete pipeline')
    full_parser.add_argument('--workflow_name', type=str, default=None, help='Workflow name')
    full_parser.add_argument('--n_samples', type=int, default=10000, help='Training samples')
    full_parser.add_argument('--batch_size', type=int, default=1000, help='Data generation batch size')
    full_parser.add_argument('--train_batch_size', type=int, default=512, help='Training batch size')
    full_parser.add_argument('--inference_samples', type=int, default=10000, help='Inference samples')
    full_parser.add_argument('--t_max', type=float, default=14.0, help='Max time (days)')
    full_parser.add_argument('--dt', type=float, default=1.0, help='Time step (days)')
    full_parser.add_argument('--observation_noise', type=float, default=1.0, help='RT-PCR noise')
    full_parser.add_argument('--detection_limit', type=float, default=-0.65, help='Detection limit')
    full_parser.add_argument('--full_trajectory', action='store_true', help='Use full trajectory')
    full_parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    full_parser.add_argument('--max_epochs', type=int, default=150, help='Max epochs')
    full_parser.add_argument('--validation_fraction', type=float, default=0.15, help='Validation fraction')
    full_parser.add_argument('--early_stopping', type=int, default=25, help='Early stopping')
    full_parser.add_argument('--hidden_features', type=int, default=256, help='Hidden features')
    full_parser.add_argument('--num_transforms', type=int, default=8, help='Number of transforms')
    full_parser.add_argument('--min_detections', type=int, default=5, help='Min detections')
    full_parser.add_argument('--min_peak_vl', type=float, default=2.0, help='Min peak viral load')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    # Initialize workflow
    workflow = TEIRVWorkflow(seed=args.seed, device=args.device)
    
    # Run requested mode
    if args.mode == 'generate':
        workflow.generate_data(args)
    elif args.mode == 'train':
        workflow.train_model(args)
    elif args.mode == 'inference':
        workflow.inference(args)
    elif args.mode == 'demo':
        workflow.demo(args)
    elif args.mode == 'full':
        workflow.full_pipeline(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()