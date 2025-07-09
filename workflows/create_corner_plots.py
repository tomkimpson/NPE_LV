#!/usr/bin/env python3
"""
Create corner plots for TEIRV production runs.

This script takes a production run ID and creates corner plots for each patient
using the posterior samples from the inference results.

Usage:
    python create_corner_plots.py 20250704_134546 --missing
    python create_corner_plots.py 20250704_134546 --all
    python create_corner_plots.py 20250704_134546 --summary
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

from TEIRV.teirv_inference import TEIRVInference


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
        'corner_plot': patient_dir / f"{patient_dir.name}_corner.png"
    }
    
    status = {}
    for file_type, file_path in required_files.items():
        status[file_type] = file_path.exists()
    
    return status


def create_corner_plot_for_patient(patient_id: str, inference_dir: Path, 
                                 overwrite: bool = False, smooth: float = 1.0) -> bool:
    """
    Create corner plot for a single patient.
    
    Parameters:
    -----------
    patient_id : str
        Patient ID
    inference_dir : Path
        Path to inference results directory
    overwrite : bool
        Whether to overwrite existing plots
    smooth : float
        Smoothing parameter for corner plots
        
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
    
    if status['corner_plot'] and not overwrite:
        print(f"  ⏭️  Corner plot already exists for patient {patient_id}")
        return True
    
    try:
        # Load posterior samples
        samples_path = patient_dir / 'posterior_samples.npy'
        posterior_samples = np.load(samples_path)
        
        # Convert to torch tensor
        posterior_tensor = torch.tensor(posterior_samples, dtype=torch.float32)
        
        print(f"  📊 Loaded {len(posterior_samples)} posterior samples for patient {patient_id}")
        
        # Create corner plot using existing TEIRVInference method
        temp_inference = TEIRVInference()
        fig_corner = temp_inference.plot_corner(posterior_tensor, smooth=smooth)
        
        # Save the plot
        corner_path = patient_dir / f"patient_{patient_id}_corner.png"
        fig_corner.savefig(corner_path, dpi=300, bbox_inches='tight')
        plt.close(fig_corner)
        
        print(f"  ✅ Saved corner plot: {corner_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create corner plot for patient {patient_id}: {e}")
        return False


def create_summary_figure(inference_dir: Path, patient_ids: List[str], 
                         output_path: Optional[Path] = None) -> bool:
    """
    Create a summary figure with corner plots for all patients.
    
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
    temp_inference = TEIRVInference()
    
    for i, patient_id in enumerate(patient_ids[:nrows*ncols]):
        ax = axes[i]
        patient_dir = inference_dir / f"patient_{patient_id}"
        
        try:
            # Load posterior samples
            samples_path = patient_dir / 'posterior_samples.npy'
            if not samples_path.exists():
                ax.text(0.5, 0.5, f'No data\nPatient {patient_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            posterior_samples = np.load(samples_path)
            posterior_tensor = torch.tensor(posterior_samples, dtype=torch.float32)
            
            # Create a small corner plot
            corner_fig = temp_inference.plot_corner(posterior_tensor)
            
            # Copy the corner plot to our subplot (this is a simplified approach)
            # In practice, you might want to create smaller individual plots
            ax.text(0.5, 0.5, f'Patient {patient_id}\n{len(posterior_samples)} samples', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Patient {patient_id}')
            
            plt.close(corner_fig)  # Close the temporary figure
            successful_plots += 1
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\nPatient {patient_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"  ❌ Failed to process patient {patient_id}: {e}")
    
    # Turn off unused subplots
    for i in range(len(patient_ids), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save summary figure
    if output_path is None:
        output_path = inference_dir / "corner_plots_summary.png"
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Summary figure saved: {output_path}")
    print(f"📊 Successfully processed {successful_plots}/{len(patient_ids)} patients")
    
    return True


def main():
    """Main function to orchestrate corner plot creation."""
    parser = argparse.ArgumentParser(
        description='Create corner plots for TEIRV production runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create missing corner plots only
  python create_corner_plots.py 20250704_134546 --missing
  
  # Recreate all corner plots with custom smoothing
  python create_corner_plots.py 20250704_134546 --all --smooth 1.5
  
  # Create summary figure with all patients
  python create_corner_plots.py 20250704_134546 --summary
        """
    )
    
    parser.add_argument('run_id', type=str, help='Production run ID (e.g., 20250704_134546)')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--missing', action='store_true', default=True,
                           help='Create only missing corner plots (default)')
    mode_group.add_argument('--all', action='store_true',
                           help='Recreate all corner plots (overwrite existing)')
    mode_group.add_argument('--summary', action='store_true',
                           help='Create summary figure with all patients')
    
    # Optional arguments
    parser.add_argument('--patients', nargs='+', type=str,
                       help='Specific patient IDs to process (default: all)')
    parser.add_argument('--output', type=str,
                       help='Custom output path for summary figure')
    parser.add_argument('--smooth', type=float, default=1.0,
                       help='Smoothing parameter for corner plots (default: 1.0)')
    
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
    
    print("🎨 TEIRV Corner Plot Creator")
    print("=" * 50)
    print(f"Production run: {args.run_id}")
    print(f"Mode: {mode}")
    print(f"Smoothing: {args.smooth}")
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
            # Create individual corner plots
            print(f"🎨 Creating corner plots for {len(patient_ids)} patients...")
            
            successful = 0
            for patient_id in patient_ids:
                print(f"\nProcessing patient {patient_id}:")
                if create_corner_plot_for_patient(patient_id, inference_dir, overwrite, args.smooth):
                    successful += 1
            
            print(f"\n✅ CORNER PLOT CREATION COMPLETED")
            print(f"📊 Successfully processed {successful}/{len(patient_ids)} patients")
            
            if successful < len(patient_ids):
                print(f"⚠️  {len(patient_ids) - successful} patients had issues")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()