#!/usr/bin/env python3
"""
Demo script for clinical NPE inference.

Quick demonstration of Phase 3: Clinical Data Integration.
Runs inference on a single patient with a small pre-trained model.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from clinical_data import ClinicalStudy
from teirv_inference import TEIRVInference
from teirv_data_generation import TEIRVDataGenerator


def quick_clinical_demo():
    """Run a quick clinical inference demo."""
    print("TEIRV NPE Clinical Inference Demo")
    print("=" * 50)
    
    # 1. Load clinical data
    print("1. Loading clinical data...")
    study = ClinicalStudy()
    
    if len(study.loader.patient_ids) == 0:
        print("❌ No clinical data found. Please check external/JSFGermano2024 repository.")
        return
    
    print(f"   Found {len(study.loader.patient_ids)} patients: {study.loader.patient_ids}")
    
    # Select a good patient for demo
    good_patients = study.filter_patients(min_detections=3, min_peak_viral_load=1.0)
    if not good_patients:
        print("❌ No suitable patients found for demo")
        return
    
    demo_patient = good_patients[0]
    print(f"   Using patient {demo_patient} for demo")
    
    # 2. Train a small NPE model
    print("\n2. Training NPE model (small dataset for demo)...")
    generator = TEIRVDataGenerator(seed=42)
    
    # Generate small training set for demo
    theta_train, x_train = generator.generate_batch(n_samples=500, batch_size=100)
    print(f"   Generated {len(theta_train)} training samples")
    
    # Train NPE
    inference = TEIRVInference(observation_type='rt_pcr', seed=123)
    
    training_info = inference.train(
        theta=theta_train,
        x=x_train,
        training_batch_size=64,
        max_num_epochs=20,  # Quick training for demo
        validation_fraction=0.2
    )
    
    print("   Training completed!")
    
    # 3. Run inference on clinical patient
    print(f"\n3. Running inference on patient {demo_patient}...")
    
    # Load patient data
    times, observations = study.loader.load_patient_data(demo_patient)
    summary = study.loader.get_patient_summary(demo_patient)
    
    print(f"   Patient summary:")
    print(f"     Timepoints: {summary['n_timepoints']}")
    print(f"     Above detection: {summary['n_above_detection']}/{summary['n_timepoints']} ({summary['detection_rate']:.1%})")
    print(f"     Peak viral load: {summary['peak_viral_load']:.2f} at day {summary['peak_time']:.1f}")
    
    # Preprocess and run inference
    x_obs = study.loader.preprocess_for_npe(demo_patient)
    posterior_samples = inference.sample_posterior(x_obs, num_samples=1000)
    
    print(f"   Sampled {len(posterior_samples)} posterior samples")
    
    # 4. Show results
    print(f"\n4. Inference results for patient {demo_patient}:")
    print("-" * 50)
    
    samples_np = posterior_samples.numpy()
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    
    print("Parameter estimates (mean ± std):")
    for i, name in enumerate(param_names):
        mean_val = np.mean(samples_np[:, i])
        std_val = np.std(samples_np[:, i])
        q025 = np.percentile(samples_np[:, i], 2.5)
        q975 = np.percentile(samples_np[:, i], 97.5)
        
        print(f"  {name}: {mean_val:7.2f} ± {std_val:5.2f} [{q025:6.2f}, {q975:6.2f}]")
    
    # 5. Quick validation
    print(f"\n5. Quick validation:")
    
    # Check if estimates are in reasonable ranges
    beta_mean = np.mean(samples_np[:, 0])
    pi_mean = np.mean(samples_np[:, 1])
    v0_mean = np.mean(samples_np[:, 5])
    
    print(f"   β (infection rate): {beta_mean:.2f} - {'✅' if 0 < beta_mean < 20 else '❓'} reasonable")
    print(f"   π (virion production): {pi_mean:.0f} - {'✅' if 200 < pi_mean < 600 else '❓'} reasonable")
    print(f"   V₀ (initial virions): {v0_mean:.0f} - {'✅' if 1 < v0_mean < 150 else '❓'} reasonable")
    
    # 6. Save demo results
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create a simple plot
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Patient data
        ax1.plot(times, observations, 'o-', linewidth=2, markersize=8, color='red')
        ax1.axhline(-0.65, color='gray', linestyle='--', alpha=0.7, label='Detection limit')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('log₁₀ Viral Load')
        ax1.set_title(f'Patient {demo_patient} RT-PCR Data')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Posterior samples (first 3 parameters)
        for i, name in enumerate(param_names[:3]):
            ax2.hist(samples_np[:, i], bins=30, alpha=0.7, label=name)
        
        ax2.set_xlabel('Parameter Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Posterior Marginals (β, π, δ)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'TEIRV NPE Demo - Patient {demo_patient}', fontsize=16)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(output_dir / f'demo_patient_{demo_patient}_{timestamp}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Demo plot saved to {output_dir}/")
        
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
    
    print(f"\n✅ Clinical inference demo completed successfully!")
    print(f"✅ Patient {demo_patient} analyzed with NPE")
    print(f"✅ Ready for full-scale clinical analysis")


if __name__ == '__main__':
    quick_clinical_demo()