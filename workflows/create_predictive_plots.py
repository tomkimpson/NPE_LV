#!/usr/bin/env python3
"""
Create predictive plots for TEIRV production runs.

This script takes a production run ID and creates predictive plots for each patient
using the posterior samples from the inference results.

Usage:
    python create_predictive_plots.py 20250704_134546 --missing
    python create_predictive_plots.py 20250704_134546 --all
    python create_predictive_plots.py 20250704_134546 --summary
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_simulator import gillespie_teirv
from TEIRV.teirv_utils import apply_observation_model, get_teirv_initial_conditions


def load_production_run_data(run_id: str) -> Path:
    """
    Load and validate production run data directory.
    
    Parameters:
    -----------
    run_id : str
        Production run ID (e.g., '20250704_134546')
        
    Returns:
    --------
    Path
        Path to the inference results directory
        
    Raises:
    -------
    FileNotFoundError
        If the production run directory doesn't exist
    """
    # Construct the expected path
    run_dir = Path(__file__).parent / f"production_run_{run_id}"
    inference_dir = run_dir / "inference_results"
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Production run directory not found: {run_dir}")
    
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference results directory not found: {inference_dir}")
    
    # Check if there are any patient directories
    patient_dirs = [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith('patient_')]
    if not patient_dirs:
        raise FileNotFoundError(f"No patient directories found in {inference_dir}")
    
    print(f"✅ Found production run: {run_dir}")
    print(f"✅ Found inference results: {inference_dir}")
    print(f"✅ Found {len(patient_dirs)} patient directories")
    
    return inference_dir


def get_patient_list(inference_dir: Path) -> List[str]:
    """
    Get list of patient IDs from the inference results directory.
    
    Parameters:
    -----------
    inference_dir : Path
        Path to inference results directory
        
    Returns:
    --------
    List[str]
        List of patient IDs
    """
    patient_dirs = [d for d in inference_dir.iterdir() if d.is_dir() and d.name.startswith('patient_')]
    patient_ids = [d.name.replace('patient_', '') for d in patient_dirs]
    return sorted(patient_ids)


def check_patient_data(patient_dir: Path) -> Dict[str, bool]:
    """
    Check if required files exist for a patient.
    
    Parameters:
    -----------
    patient_dir : Path
        Path to patient directory
        
    Returns:
    --------
    Dict[str, bool]
        Status of required files
    """
    required_files = {
        'posterior_samples': patient_dir / 'posterior_samples.npy',
        'observations': patient_dir / 'observations.npy',
        'predictive_plot': patient_dir / f"{patient_dir.name}_predictive.png"
    }
    
    status = {}
    for file_type, file_path in required_files.items():
        status[file_type] = file_path.exists()
    
    return status


def create_predictive_plot_for_patient(patient_id: str, inference_dir: Path, 
                                     overwrite: bool = False) -> bool:
    """
    Create predictive plot for a single patient using posterior mean prediction.
    
    Parameters:
    -----------
    patient_id : str
        Patient ID
    inference_dir : Path
        Path to inference results directory
    overwrite : bool
        Whether to overwrite existing plots
        
    Returns:
    --------
    bool
        True if plot was created successfully
    """
    patient_dir = inference_dir / f"patient_{patient_id}"
    
    if not patient_dir.exists():
        print(f"  ❌ Patient directory not found: {patient_dir}")
        return False
    
    # Check if files exist
    status = check_patient_data(patient_dir)
    
    if not status['posterior_samples']:
        print(f"  ❌ Missing posterior samples for patient {patient_id}")
        return False
    
    if not status['observations']:
        print(f"  ❌ Missing observations for patient {patient_id}")
        return False
    
    if status['predictive_plot'] and not overwrite:
        print(f"  ⏭️  Predictive plot already exists for patient {patient_id}")
        return True
    
    try:
        # Load posterior samples and observations
        samples_path = patient_dir / 'posterior_samples.npy'
        observations_path = patient_dir / 'observations.npy'
        
        posterior_samples = np.load(samples_path)
        observations = np.load(observations_path)
        
        # Convert to torch tensors
        posterior_tensor = torch.tensor(posterior_samples, dtype=torch.float32)
        observations_tensor = torch.tensor(observations, dtype=torch.float32)
        
        print(f"  📊 Loaded {len(posterior_samples)} posterior samples for patient {patient_id}")
        print(f"  📊 Loaded {len(observations)} observations for patient {patient_id}")
        
        # Create predictive plot
        fig = _create_predictive_plot(patient_id, posterior_tensor, observations_tensor)
        
        # Save the plot
        predictive_path = patient_dir / f"patient_{patient_id}_predictive.png"
        fig.savefig(predictive_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✅ Saved predictive plot: {predictive_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create predictive plot for patient {patient_id}: {e}")
        return False


def _create_predictive_plot(patient_id: str, posterior_samples: torch.Tensor, 
                           observations: torch.Tensor) -> plt.Figure:
    """
    Create predictive plot using posterior mean prediction.
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier
    posterior_samples : torch.Tensor
        Posterior parameter samples
    observations : torch.Tensor
        Observed RT-PCR data
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Predictive plot figure
    """
    print(f"  Generating prediction using posterior mean...")
    
    # Timing for progress estimation
    import time
    start_time = time.time()
    
    # Time grids with higher resolution for smoother curves
    t_obs = np.arange(0, 11, 1.0)  # Observed range: 0-10 days (training data)
    t_pred = np.arange(0, 21, 0.1)  # Extended range: 0-20 days with 0.1 day steps
    
    # Compute posterior mean parameters
    posterior_mean = torch.mean(posterior_samples, dim=0)
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    print(f"    Using posterior mean parameters:")
    for i, (name, value) in enumerate(zip(param_names, posterior_mean.numpy())):
        print(f"      {name}: {value:.4f}")
    
    # Integration timesteps info
    n_timesteps = len(t_pred)
    timestep_size = t_pred[1] - t_pred[0]
    print(f"    Integration timesteps: {n_timesteps} steps (Δt = {timestep_size} days)")
    
    # Generate single prediction using posterior mean
    base_ic = get_teirv_initial_conditions()
    
    try:
        # Set up initial conditions
        ic = base_ic.copy()
        ic['V'] = posterior_mean[5].item()  # V₀ from posterior mean
        
        # Simulate for extended range with fine time grid
        _, trajectory_ext = gillespie_teirv(
            theta=posterior_mean.numpy(),
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
            add_noise=False  # No noise for mean prediction
        )
        
    except Exception as e:
        print(f"  ❌ Simulation failed: {e}")
        return plt.figure()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot observed data points
    ax.scatter(t_obs, observations.numpy(), color='black', s=80, 
              label='Observed data', zorder=10, alpha=0.8)
    
    # Plot mean prediction curve
    ax.plot(t_pred, obs_ext, color='blue', linewidth=2, 
           label='Posterior mean prediction')
    
    # Add vertical line at transition
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.7, 
              label='Training/Prediction boundary')
    
    # Formatting
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('log₁₀ Viral Load', fontsize=12)
    ax.set_title(f'Patient {patient_id}: Posterior Mean Prediction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add parameter info as text box
    param_text = f"Parameters (posterior mean):\n"
    for i, (name, value) in enumerate(zip(param_names, posterior_mean.numpy())):
        param_text += f"{name}: {value:.4f}\n"
    param_text += f"Integration: {n_timesteps} steps (Δt={timestep_size})"
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    # Set reasonable y-limits for log10 viral load data
    all_data = np.concatenate([observations.numpy(), obs_ext])
    y_min = np.min(all_data) - 0.5  # Add some padding below
    y_max = np.max(all_data) + 0.5  # Add some padding above
    ax.set_ylim(y_min, y_max)
    
    # Add horizontal line for detection limit
    ax.axhline(y=-0.65, color='gray', linestyle=':', alpha=0.7, 
              label='Detection limit')
    
    plt.tight_layout()
    
    simulation_time = time.time() - start_time
    print(f"  ✅ Generated posterior mean prediction in {simulation_time:.1f}s")
    return fig


def create_summary_figure(inference_dir: Path, patient_ids: List[str], 
                         output_path: Optional[Path] = None) -> bool:
    """
    Create a summary figure with predictive plots for all patients.
    
    Parameters:
    -----------
    inference_dir : Path
        Path to inference results directory
    patient_ids : List[str]
        List of patient IDs to include
    output_path : Optional[Path]
        Custom output path for summary figure
        
    Returns:
    --------
    bool
        True if summary was created successfully
    """
    print(f"📊 Creating summary figure with {len(patient_ids)} patients...")
    
    # Determine grid size
    n_patients = len(patient_ids)
    if n_patients == 1:
        nrows, ncols = 1, 1
    elif n_patients <= 4:
        nrows, ncols = 2, 2
    elif n_patients <= 6:
        nrows, ncols = 2, 3
    elif n_patients <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 4  # Maximum grid size
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    successful_plots = 0
    
    for i, patient_id in enumerate(patient_ids[:nrows*ncols]):
        ax = axes[i]
        patient_dir = inference_dir / f"patient_{patient_id}"
        
        try:
            # Load posterior samples and observations
            samples_path = patient_dir / 'posterior_samples.npy'
            observations_path = patient_dir / 'observations.npy'
            
            if not samples_path.exists() or not observations_path.exists():
                ax.text(0.5, 0.5, f'No data\\nPatient {patient_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            posterior_samples = np.load(samples_path)
            observations = np.load(observations_path)
            
            posterior_tensor = torch.tensor(posterior_samples, dtype=torch.float32)
            observations_tensor = torch.tensor(observations, dtype=torch.float32)
            
            # Create a compact predictive plot
            t_obs = np.arange(0, 11, 1.0)
            t_pred = np.arange(0, 21, 0.5)  # Coarser grid for summary
            
            # Compute posterior mean
            posterior_mean = torch.mean(posterior_tensor, dim=0)
            
            # Generate prediction
            base_ic = get_teirv_initial_conditions()
            ic = base_ic.copy()
            ic['V'] = posterior_mean[5].item()
            
            _, trajectory_ext = gillespie_teirv(
                theta=posterior_mean.numpy(),
                initial_conditions=ic,
                t_max=20.0,
                t_grid=t_pred,
                max_steps=1000000
            )
            
            V_trajectory_ext = trajectory_ext[:, 4]
            obs_ext = apply_observation_model(
                V_trajectory=V_trajectory_ext,
                sigma=1.0,
                detection_limit=-0.65,
                add_noise=False
            )
            
            # Plot on subplot
            ax.scatter(t_obs, observations_tensor.numpy(), color='black', s=40, alpha=0.8)
            ax.plot(t_pred, obs_ext, color='blue', linewidth=1)
            ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'Patient {patient_id}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set reasonable limits
            all_data = np.concatenate([observations_tensor.numpy(), obs_ext])
            y_min = np.min(all_data) - 0.5
            y_max = np.max(all_data) + 0.5
            ax.set_ylim(y_min, y_max)
            
            successful_plots += 1
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\\nPatient {patient_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"  ❌ Failed to process patient {patient_id}: {e}")
    
    # Turn off unused subplots
    for i in range(len(patient_ids), len(axes)):
        axes[i].axis('off')
    
    # Add overall labels
    fig.text(0.5, 0.02, 'Time (days)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'log₁₀ Viral Load', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout()
    
    # Save summary figure
    if output_path is None:
        output_path = inference_dir / "predictive_plots_summary.png"
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Summary figure saved: {output_path}")
    print(f"📊 Successfully processed {successful_plots}/{len(patient_ids)} patients")
    
    return True


def main():
    """Main function to orchestrate predictive plot creation."""
    parser = argparse.ArgumentParser(
        description='Create predictive plots for TEIRV production runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create missing predictive plots only
  python create_predictive_plots.py 20250704_134546 --missing
  
  # Recreate all predictive plots
  python create_predictive_plots.py 20250704_134546 --all
  
  # Create summary figure with all patients
  python create_predictive_plots.py 20250704_134546 --summary
        """
    )
    
    parser.add_argument('run_id', type=str, help='Production run ID (e.g., 20250704_134546)')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--missing', action='store_true', default=True,
                           help='Create only missing predictive plots (default)')
    mode_group.add_argument('--all', action='store_true',
                           help='Recreate all predictive plots (overwrite existing)')
    mode_group.add_argument('--summary', action='store_true',
                           help='Create summary figure with all patients')
    
    # Optional arguments
    parser.add_argument('--patients', nargs='+', type=str,
                       help='Specific patient IDs to process (default: all)')
    parser.add_argument('--output', type=str,
                       help='Custom output path for summary figure')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.all:
        mode = 'all'
        overwrite = True
    elif args.summary:
        mode = 'summary'
        overwrite = False
    else:
        mode = 'missing'
        overwrite = False
    
    print("📈 TEIRV Predictive Plot Creator")
    print("=" * 50)
    print(f"Production run: {args.run_id}")
    print(f"Mode: {mode}")
    print()
    
    try:
        # Load and validate production run data
        inference_dir = load_production_run_data(args.run_id)
        
        # Get patient list
        if args.patients:
            patient_ids = args.patients
            print(f"📋 Processing specified patients: {patient_ids}")
        else:
            patient_ids = get_patient_list(inference_dir)
            print(f"📋 Processing all patients: {patient_ids}")
        
        if not patient_ids:
            print("❌ No patients to process")
            return
        
        # Process based on mode
        if mode == 'summary':
            # Create summary figure
            output_path = Path(args.output) if args.output else None
            create_summary_figure(inference_dir, patient_ids, output_path)
        else:
            # Create individual predictive plots
            print(f"📈 Creating predictive plots for {len(patient_ids)} patients...")
            
            successful = 0
            for patient_id in patient_ids:
                print(f"\\nProcessing patient {patient_id}:")
                if create_predictive_plot_for_patient(patient_id, inference_dir, overwrite):
                    successful += 1
            
            print(f"\\n✅ PREDICTIVE PLOT CREATION COMPLETED")
            print(f"📊 Successfully processed {successful}/{len(patient_ids)} patients")
            
            if successful < len(patient_ids):
                print(f"⚠️  {len(patient_ids) - successful} patients had issues")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()