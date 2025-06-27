#!/usr/bin/env python3
"""
Run NPE inference on clinical RT-PCR data from COVID patients.

This script demonstrates Phase 3 of the TEIRV NPE implementation:
integrating real clinical data from the JSFGermano2024 repository.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from clinical_data import ClinicalStudy, load_clinical_validation_data, validate_clinical_data_compatibility
from teirv_inference import TEIRVInference
from teirv_data_generation import TEIRVDataGenerator
from teirv_utils import teirv_parameter_summary


def load_trained_model(model_path: str = None) -> TEIRVInference:
    """
    Load a pre-trained NPE model or train a new one.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to saved model. If None, trains a new model.
        
    Returns:
    --------
    inference : TEIRVInference
        Trained inference object
    """
    if model_path and Path(model_path).exists():
        print(f"Loading pre-trained model from {model_path}")
        inference = TEIRVInference.load(model_path)
    else:
        print("Training new NPE model on synthetic data...")
        
        # Generate training data
        generator = TEIRVDataGenerator(seed=42)
        print("Generating training data (this may take a few minutes)...")
        
        theta_train, x_train = generator.generate_batch(
            n_samples=5000,  # Reasonable size for clinical inference
            batch_size=500
        )
        
        print(f"Generated {len(theta_train)} training samples")
        
        # Train NPE
        inference = TEIRVInference(observation_type='rt_pcr', seed=123)
        
        training_info = inference.train(
            theta=theta_train,
            x=x_train,
            training_batch_size=128,
            max_num_epochs=50,
            validation_fraction=0.2
        )
        
        print("Training completed successfully")
        
        # Save model for future use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/teirv_npe_clinical_{timestamp}.pkl"
        Path("models").mkdir(exist_ok=True)
        inference.save(save_path)
        print(f"Model saved to {save_path}")
    
    return inference


def run_patient_inference(inference: TEIRVInference, 
                         patient_id: str,
                         study: ClinicalStudy,
                         num_samples: int = 10000) -> dict:
    """
    Run NPE inference for a specific patient.
    
    Parameters:
    -----------
    inference : TEIRVInference
        Trained NPE model
    patient_id : str
        Patient ID to analyze
    study : ClinicalStudy
        Clinical study object
    num_samples : int
        Number of posterior samples to generate
        
    Returns:
    --------
    results : dict
        Inference results including samples and summary
    """
    print(f"\nRunning inference for patient {patient_id}...")
    
    # Load patient data
    times, observations = study.loader.load_patient_data(patient_id)
    summary = study.loader.get_patient_summary(patient_id)
    
    print(f"Patient {patient_id} summary:")
    print(f"  Timepoints: {summary['n_timepoints']}")
    print(f"  Above detection: {summary['n_above_detection']}/{summary['n_timepoints']} ({summary['detection_rate']:.1%})")
    print(f"  Peak viral load: {summary['peak_viral_load']:.2f} at day {summary['peak_time']:.1f}")
    print(f"  Viral load range: [{summary['viral_load_range'][0]:.2f}, {summary['viral_load_range'][1]:.2f}]")
    
    # Preprocess for NPE
    x_obs = study.loader.preprocess_for_npe(patient_id)
    print(f"Preprocessed observations shape: {x_obs.shape}")
    
    # Sample from posterior
    print(f"Sampling {num_samples} from posterior...")
    posterior_samples = inference.sample_posterior(x_obs, num_samples=num_samples)
    
    # Compute summary statistics
    param_summary = teirv_parameter_summary(posterior_samples)
    
    # Print results
    print("\nPosterior parameter estimates:")
    print("-" * 60)
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    for name in param_names:
        stats = param_summary[name]
        print(f"{name:>3}: {stats['mean']:8.2f} ± {stats['std']:6.2f} "
              f"[{stats['q025']:6.2f}, {stats['q975']:6.2f}]")
    
    results = {
        'patient_id': patient_id,
        'times': times,
        'observations': observations,
        'summary': summary,
        'posterior_samples': posterior_samples,
        'param_summary': param_summary,
        'x_obs': x_obs
    }
    
    return results


def create_patient_plots(inference: TEIRVInference, 
                        results: dict,
                        output_dir: Path) -> None:
    """
    Create visualization plots for patient inference results.
    
    Parameters:
    -----------
    inference : TEIRVInference
        Trained NPE model
    results : dict
        Patient inference results
    output_dir : Path
        Output directory for plots
    """
    patient_id = results['patient_id']
    posterior_samples = results['posterior_samples']
    observations = results['observations']
    
    print(f"Creating plots for patient {patient_id}...")
    
    # 1. Posterior marginals
    try:
        fig1 = inference.plot_posterior_samples(posterior_samples)
        fig1.suptitle(f'Patient {patient_id} - Posterior Marginals', fontsize=16)
        fig1.savefig(output_dir / f'patient_{patient_id}_marginals.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  ✅ Saved marginals plot")
    except Exception as e:
        print(f"  ❌ Marginals plot failed: {e}")
    
    # 2. Pairwise posterior relationships
    try:
        fig2 = inference.plot_pairwise(posterior_samples)
        fig2.suptitle(f'Patient {patient_id} - Posterior Pairwise', fontsize=16)
        fig2.savefig(output_dir / f'patient_{patient_id}_pairwise.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✅ Saved pairwise plot")
    except Exception as e:
        print(f"  ❌ Pairwise plot failed: {e}")
    
    # 3. Corner plot (if available)
    try:
        fig3 = inference.plot_corner(posterior_samples)
        fig3.suptitle(f'Patient {patient_id} - Posterior Corner Plot', fontsize=16)
        fig3.savefig(output_dir / f'patient_{patient_id}_corner.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  ✅ Saved corner plot")
    except Exception as e:
        print(f"  ❌ Corner plot failed: {e}")
    
    # 4. Posterior predictive check
    try:
        x_obs = results['x_obs']
        fig4 = inference.posterior_predictive_check(posterior_samples, x_obs)
        fig4.suptitle(f'Patient {patient_id} - Posterior Predictive Check', fontsize=16)
        fig4.savefig(output_dir / f'patient_{patient_id}_predictive.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  ✅ Saved predictive check plot")
    except Exception as e:
        print(f"  ❌ Predictive check failed: {e}")
    
    # 5. Raw data plot
    try:
        fig5, ax = plt.subplots(figsize=(10, 6))
        times = results['times']
        
        ax.plot(times, observations, 'o-', linewidth=2, markersize=8, 
               color='red', label='RT-PCR observations')
        ax.axhline(-0.65, color='gray', linestyle='--', alpha=0.7, 
                  label='Detection limit')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('log₁₀ Viral Load')
        ax.set_title(f'Patient {patient_id} - Clinical RT-PCR Data')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add summary stats
        summary = results['summary']
        textstr = (f"Peak: {summary['peak_viral_load']:.2f} at day {summary['peak_time']:.1f}\n"
                  f"Detection rate: {summary['detection_rate']:.1%}")
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        fig5.savefig(output_dir / f'patient_{patient_id}_raw_data.png', 
                    dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print(f"  ✅ Saved raw data plot")
    except Exception as e:
        print(f"  ❌ Raw data plot failed: {e}")


def main():
    """Main clinical inference workflow."""
    parser = argparse.ArgumentParser(description='Run NPE inference on clinical COVID RT-PCR data')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pre-trained NPE model')
    parser.add_argument('--patient-id', type=str, default=None,
                       help='Specific patient ID to analyze (if None, analyzes all good patients)')
    parser.add_argument('--min-detections', type=int, default=5,
                       help='Minimum detections above limit for patient filtering')
    parser.add_argument('--min-peak-vl', type=float, default=2.0,
                       help='Minimum peak viral load for patient filtering')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of posterior samples to generate')
    parser.add_argument('--output-dir', type=str, default='clinical_results',
                       help='Output directory for results')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate clinical data compatibility')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEIRV NPE CLINICAL INFERENCE")
    print("=" * 80)
    
    # Validate clinical data
    if not validate_clinical_data_compatibility():
        print("❌ Clinical data validation failed. Check repository setup.")
        return
    
    if args.validate_only:
        print("✅ Clinical data validation completed.")
        return
    
    # Load clinical study
    print("\nLoading clinical study...")
    study = ClinicalStudy()
    
    if len(study.loader.patient_ids) == 0:
        print("❌ No patient data found. Check external/JSFGermano2024 repository.")
        return
    
    # Filter patients by quality
    if args.patient_id:
        patient_ids = [args.patient_id] if args.patient_id in study.loader.patient_ids else []
        if not patient_ids:
            print(f"❌ Patient {args.patient_id} not found in dataset")
            return
    else:
        patient_ids = study.filter_patients(
            min_detections=args.min_detections,
            min_peak_viral_load=args.min_peak_vl
        )
    
    if not patient_ids:
        print("❌ No patients meet the quality criteria")
        return
    
    print(f"Analyzing {len(patient_ids)} patient(s): {patient_ids}")
    
    # Load or train NPE model
    inference = load_trained_model(args.model_path)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"clinical_inference_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Run inference for each patient
    all_results = {}
    
    for patient_id in patient_ids:
        try:
            results = run_patient_inference(
                inference=inference,
                patient_id=patient_id,
                study=study,
                num_samples=args.num_samples
            )
            
            all_results[patient_id] = results
            
            # Create plots for this patient
            patient_output_dir = output_dir / f"patient_{patient_id}"
            patient_output_dir.mkdir(exist_ok=True)
            
            create_patient_plots(inference, results, patient_output_dir)
            
            # Save numerical results
            param_summary = results['param_summary']
            with open(patient_output_dir / f"patient_{patient_id}_summary.txt", 'w') as f:
                f.write(f"Patient {patient_id} NPE Inference Results\n")
                f.write("=" * 50 + "\n\n")
                
                # Patient summary
                summary = results['summary']
                f.write("Clinical Data Summary:\n")
                f.write(f"  Timepoints: {summary['n_timepoints']}\n")
                f.write(f"  Above detection: {summary['n_above_detection']}/{summary['n_timepoints']} ({summary['detection_rate']:.1%})\n")
                f.write(f"  Peak viral load: {summary['peak_viral_load']:.2f} at day {summary['peak_time']:.1f}\n")
                f.write(f"  Viral load range: [{summary['viral_load_range'][0]:.2f}, {summary['viral_load_range'][1]:.2f}]\n\n")
                
                # Parameter estimates
                f.write("Posterior Parameter Estimates:\n")
                f.write("-" * 40 + "\n")
                f.write("Param    Mean   ±   Std     [2.5%,  97.5%]\n")
                f.write("-" * 40 + "\n")
                
                param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
                for name in param_names:
                    stats = param_summary[name]
                    f.write(f"{name:>5}: {stats['mean']:7.2f} ± {stats['std']:6.2f} "
                           f"[{stats['q025']:6.2f}, {stats['q975']:6.2f}]\n")
            
            print(f"✅ Completed analysis for patient {patient_id}")
            
        except Exception as e:
            print(f"❌ Failed to analyze patient {patient_id}: {e}")
            continue
    
    # Create summary report
    if all_results:
        print(f"\n✅ Successfully analyzed {len(all_results)} patients")
        print(f"✅ Results saved to {output_dir}")
        
        # Create comparison plot if multiple patients
        if len(all_results) > 1:
            try:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
                
                for i, param in enumerate(param_names):
                    ax = axes[i]
                    
                    means = []
                    stds = []
                    labels = []
                    
                    for patient_id, results in all_results.items():
                        stats = results['param_summary'][param]
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                        labels.append(f"P{patient_id}")
                    
                    x_pos = range(len(means))
                    ax.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(labels, rotation=45)
                    ax.set_ylabel(f'{param}')
                    ax.set_title(f'Parameter {param} Across Patients')
                    ax.grid(True, alpha=0.3)
                
                plt.suptitle('Parameter Estimates Across All Patients', fontsize=16)
                plt.tight_layout()
                plt.savefig(output_dir / 'all_patients_comparison.png', 
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"✅ Saved multi-patient comparison plot")
                
            except Exception as e:
                print(f"❌ Multi-patient comparison plot failed: {e}")
    
    else:
        print("❌ No patients were successfully analyzed")
    
    print("\n" + "=" * 80)
    print("CLINICAL INFERENCE COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()