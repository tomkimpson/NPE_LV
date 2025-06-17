#!/usr/bin/env python3
"""
Clinical inference script for TEIRV NPE.

Performs NPE inference on clinical RT-PCR data from JSFGermano2024 study.
For each patient:
1. Load RT-PCR time series
2. Use trained NPE to sample posterior  
3. Generate parameter estimates and credible intervals
4. Save results for comparison with particle filter benchmark
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.clinical_data import ClinicalStudy, ClinicalDataLoader
from TEIRV.teirv_inference import TEIRVInference
from TEIRV.teirv_utils import create_teirv_time_grid, teirv_parameter_summary


class ClinicalInferenceResults:
    """Container for clinical inference results."""
    
    def __init__(self):
        self.patient_results = {}
        self.summary_stats = {}
        self.inference_time = {}
        
    def add_patient_result(self, patient_id: str, 
                          posterior_samples: torch.Tensor,
                          observations: torch.Tensor,
                          inference_time: float):
        """Add results for a patient."""
        self.patient_results[patient_id] = {
            'posterior_samples': posterior_samples,
            'observations': observations,
            'parameter_summary': teirv_parameter_summary(posterior_samples)
        }
        self.inference_time[patient_id] = inference_time
        
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary statistics as DataFrame."""
        rows = []
        
        for patient_id, results in self.patient_results.items():
            summary = results['parameter_summary']
            row = {'patient_id': patient_id}
            
            # Add parameter estimates (means and credible intervals)
            for param, stats in summary.items():
                row[f'{param}_mean'] = stats['mean']
                row[f'{param}_q025'] = stats['q025']
                row[f'{param}_q975'] = stats['q975']
                
            # Add inference metadata
            row['inference_time'] = self.inference_time[patient_id]
            row['n_samples'] = len(results['posterior_samples'])
            
            rows.append(row)
            
        return pd.DataFrame(rows)
        
    def save_results(self, output_dir: str):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary DataFrame
        summary_df = self.get_summary_dataframe()
        summary_df.to_csv(output_path / 'clinical_parameter_estimates.csv', index=False)
        
        # Save individual patient results
        for patient_id, results in self.patient_results.items():
            patient_dir = output_path / f'patient_{patient_id}'
            patient_dir.mkdir(exist_ok=True)
            
            # Save posterior samples
            samples = results['posterior_samples'].numpy()
            np.save(patient_dir / 'posterior_samples.npy', samples)
            
            # Save observations
            obs = results['observations'].numpy()
            np.save(patient_dir / 'observations.npy', obs)
            
            # Save parameter summary
            summary = results['parameter_summary']
            summary_df = pd.DataFrame(summary).T
            summary_df.to_csv(patient_dir / 'parameter_summary.csv')
            
        print(f"Results saved to {output_path}")


def load_trained_npe_model(model_path: str) -> TEIRVInference:
    """Load trained NPE model."""
    print(f"Loading trained NPE model from {model_path}")
    
    # Initialize inference object
    teirv_npe = TEIRVInference()
    
    # Load the trained model
    teirv_npe.load_model(model_path)
    
    print("✅ NPE model loaded successfully")
    return teirv_npe


def perform_patient_inference(npe_model: TEIRVInference, 
                            patient_id: str,
                            observations: torch.Tensor,
                            n_samples: int = 10000) -> Tuple[torch.Tensor, float]:
    """
    Perform NPE inference for a single patient.
    
    Parameters:
    -----------
    npe_model : TEIRVInference
        Trained NPE model
    patient_id : str
        Patient identifier
    observations : torch.Tensor
        Patient RT-PCR observations
    n_samples : int
        Number of posterior samples
        
    Returns:
    --------
    posterior_samples : torch.Tensor
        Samples from posterior distribution
    inference_time : float
        Time taken for inference (seconds)
    """
    print(f"\nPerforming inference for patient {patient_id}")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Observation range: [{observations.min():.2f}, {observations.max():.2f}]")
    
    start_time = time.time()
    
    try:
        # Sample from posterior
        posterior_samples = npe_model.sample_posterior(
            observations.unsqueeze(0),  # Add batch dimension
            n_samples=n_samples
        )
        
        inference_time = time.time() - start_time
        
        print(f"  ✅ Inference completed in {inference_time:.2f}s")
        print(f"  Posterior samples shape: {posterior_samples.shape}")
        
        return posterior_samples.squeeze(0), inference_time  # Remove batch dimension
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        raise


def create_clinical_summary_plots(results: ClinicalInferenceResults, 
                                output_dir: str):
    """Create summary plots for clinical results."""
    output_path = Path(output_dir)
    
    # Get summary dataframe
    summary_df = results.get_summary_dataframe()
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    
    # 1. Parameter estimates with credible intervals
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Extract data
        means = summary_df[f'{param}_mean']
        lower = summary_df[f'{param}_q025'] 
        upper = summary_df[f'{param}_q975']
        patients = summary_df['patient_id']
        
        # Plot means with error bars
        ax.errorbar(range(len(patients)), means, 
                   yerr=[means - lower, upper - means],
                   fmt='o', capsize=5, capthick=2)
        
        ax.set_xlabel('Patient')
        ax.set_ylabel(f'{param}')
        ax.set_title(f'Parameter {param} Estimates')
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels([p[:6] for p in patients], rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_estimates_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual patient posterior distributions
    n_patients = len(results.patient_results)
    fig, axes = plt.subplots(n_patients, len(param_names), 
                            figsize=(20, 4*n_patients))
    
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    for i, (patient_id, patient_results) in enumerate(results.patient_results.items()):
        samples = patient_results['posterior_samples'].numpy()
        
        for j, param in enumerate(param_names):
            ax = axes[i, j]
            
            # Plot histogram
            ax.hist(samples[:, j], bins=50, alpha=0.7, density=True)
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            
            if i == 0:
                ax.set_title(f'{param}')
            if j == 0:
                ax.text(-0.15, 0.5, f'Patient {patient_id[:6]}', 
                       transform=ax.transAxes, rotation=90, 
                       va='center', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Perform NPE inference on clinical TEIRV data')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained NPE model (.pkl file)')
    parser.add_argument('--output', type=str, default='results/clinical_inference',
                       help='Output directory for results')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of posterior samples per patient')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                       help='Specific patient IDs to analyze (default: all)')
    parser.add_argument('--min_detections', type=int, default=5,
                       help='Minimum detections for patient inclusion')
    parser.add_argument('--min_peak_vl', type=float, default=2.0,
                       help='Minimum peak viral load for inclusion')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to clinical data directory')
    
    args = parser.parse_args()
    
    print("TEIRV Clinical Inference Pipeline")
    print("=" * 50)
    
    # 1. Load trained NPE model
    npe_model = load_trained_npe_model(args.model)
    
    # 2. Load clinical data
    print("\nLoading clinical data...")
    study = ClinicalStudy(args.data_dir)
    
    # Get study summary
    summary_df = study.get_study_summary()
    print(f"Found {len(study.loader.patient_ids)} patients")
    
    # 3. Filter patients based on data quality
    if args.patients is None:
        print(f"\nFiltering patients (min_detections={args.min_detections}, min_peak_vl={args.min_peak_vl})")
        good_patients = study.filter_patients(
            min_detections=args.min_detections,
            min_peak_viral_load=args.min_peak_vl
        )
    else:
        good_patients = args.patients
        print(f"Using specified patients: {good_patients}")
    
    if not good_patients:
        print("❌ No patients meet inclusion criteria")
        return
        
    # 4. Prepare data for inference
    print(f"\nPreparing {len(good_patients)} patients for inference...")
    
    # Use standard TEIRV time grid
    time_grid = create_teirv_time_grid(14.0, 1.0)
    patient_data = study.prepare_for_inference(good_patients, time_grid)
    
    # 5. Perform inference for each patient
    results = ClinicalInferenceResults()
    
    print(f"\nPerforming NPE inference ({args.n_samples} samples per patient)...")
    total_start = time.time()
    
    for patient_id in good_patients:
        if patient_id not in patient_data:
            print(f"⚠️  Skipping patient {patient_id} (data preparation failed)")
            continue
            
        observations = patient_data[patient_id]
        
        try:
            posterior_samples, inference_time = perform_patient_inference(
                npe_model, patient_id, observations, args.n_samples
            )
            
            results.add_patient_result(
                patient_id, posterior_samples, observations, inference_time
            )
            
        except Exception as e:
            print(f"❌ Failed to process patient {patient_id}: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # 6. Generate summary and save results
    print(f"\nInference completed in {total_time:.2f}s")
    print(f"Successfully processed {len(results.patient_results)} patients")
    
    # Save results
    print(f"\nSaving results to {args.output}")
    results.save_results(args.output)
    
    # Create summary plots
    print("Creating summary plots...")
    create_clinical_summary_plots(results, args.output)
    
    # Print parameter summary
    summary_df = results.get_summary_dataframe()
    print("\nParameter Estimates Summary:")
    print("=" * 80)
    
    # Show key parameters
    key_cols = ['patient_id', 'β_mean', 'π_mean', 'δ_mean', 'φ_mean', 'ρ_mean', 'V₀_mean']
    if all(col in summary_df.columns for col in key_cols):
        print(summary_df[key_cols].round(3).to_string(index=False))
    
    print(f"\n✅ Clinical inference pipeline completed successfully")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()