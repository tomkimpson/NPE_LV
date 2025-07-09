#!/usr/bin/env python3
"""
Create parameter posterior grid plots for TEIRV production runs.

This script creates a grid plot showing the 1D posterior distributions for each 
TEIRV parameter across all patients. Each row corresponds to a patient, each column 
to a parameter, with shared x-axis domains for each parameter column.

Usage:
    python create_parameter_posteriors_grid.py 20250704_134546
    python create_parameter_posteriors_grid.py 20250704_134546 --patients 432192 443108
    python create_parameter_posteriors_grid.py 20250704_134546 --output /path/to/output.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from scipy.stats import gaussian_kde
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


def load_parameter_data(inference_dir: Path, patient_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Load posterior parameter samples for all patients.
    
    Parameters:
    -----------
    inference_dir : Path
        Path to inference results directory
    patient_ids : List[str]
        List of patient IDs to process
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping patient IDs to parameter sample arrays of shape (n_samples, 6)
        Parameters are in order: [β, π, δ, φ, ρ, V₀]
    """
    parameter_data = {}
    
    for patient_id in patient_ids:
        patient_dir = inference_dir / f"patient_{patient_id}"
        samples_path = patient_dir / 'posterior_samples.npy'
        
        if not samples_path.exists():
            print(f"⚠️  Missing posterior samples for patient {patient_id}")
            continue
            
        try:
            # Load posterior samples
            posterior_samples = np.load(samples_path)
            parameter_data[patient_id] = posterior_samples
            print(f"📊 Loaded {len(posterior_samples)} samples for patient {patient_id}")
            
        except Exception as e:
            print(f"❌ Failed to load data for patient {patient_id}: {e}")
            continue
    
    return parameter_data


def create_parameter_grid_plot(parameter_data: Dict[str, np.ndarray], 
                              output_path: Optional[Path] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              smooth: float = 1.0) -> Path:
    """
    Create parameter posterior grid plot for all patients.
    
    Parameters:
    -----------
    parameter_data : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to parameter sample arrays
    output_path : Optional[Path]
        Custom output path for the plot
    figsize : Optional[Tuple[int, int]]
        Custom figure size (width, height)
    smooth : float
        Smoothing parameter for density estimation (default: 1.0)
        
    Returns:
    --------
    Path
        Path to the saved plot
    """
    if not parameter_data:
        raise ValueError("No parameter data available for plotting")
    
    # Parameter configuration - matches corner plot ordering
    param_names = ['β', 'ρ', 'π', 'φ', 'δ', 'log₁₀V₀']
    param_labels = [r'$\beta$', r'$\rho$', r'$\pi$', r'$\phi$', r'$\delta$', r'$\log_{10}V_0$']
    
    # Parameter bounds from TEIRV prior for consistent x-axis limits (updated ordering)
    param_bounds = [
        (0.0, 20.0),      # β: infection rate
        (0.0, 1.0),       # ρ: reversion rate
        (200.0, 600.0),   # π: virion production
        (0.0, 15.0),      # φ: interferon protection
        (1.0, 11.0),      # δ: cell clearance
        (0.0, 5.0),       # log₁₀V₀: log initial viral load
    ]
    
    # Sort patient IDs for consistent ordering
    patient_order = sorted(parameter_data.keys())
    n_patients = len(patient_order)
    n_params = len(param_names)
    
    # Determine figure size
    if figsize is None:
        fig_width = max(15, n_params * 3)
        fig_height = max(8, n_patients * 2)
        figsize = (fig_width, fig_height)
    
    # Create subplot grid with shared x-axis per column
    fig, axes = plt.subplots(n_patients, n_params, figsize=figsize, 
                            sharex='col', sharey=False)
    
    # Handle single patient case
    if n_patients == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for each patient
    colors = plt.cm.Set1(np.linspace(0, 1, min(n_patients, 9)))
    
    # Create TEIRVInference instance for parameter transformation
    temp_inference = TEIRVInference()
    
    # Plot each patient's parameter posteriors
    for i, patient_id in enumerate(patient_order):
        samples = parameter_data[patient_id]
        color = colors[i % len(colors)]
        
        # Transform samples to match corner plot ordering and log₁₀V₀
        samples_tensor = torch.tensor(samples, dtype=torch.float32)
        transformed_samples = temp_inference._prepare_samples_for_display(samples_tensor)
        transformed_samples_np = transformed_samples.numpy()
        
        for j, (param_name, param_bounds_j) in enumerate(zip(param_names, param_bounds)):
            ax = axes[i, j]
            param_samples = transformed_samples_np[:, j]
            
            # Create smooth density using corner approach
            try:
                # Create a temporary corner plot for this patient to extract smooth density
                corner_fig = temp_inference.plot_corner(samples_tensor, smooth=smooth)
                
                # Extract density data from the diagonal subplot
                # Corner plots have diagonal elements at positions (j, j)
                corner_axes = corner_fig.get_axes()
                n_params_corner = int(np.sqrt(len(corner_axes)))
                diag_idx = j * n_params_corner + j
                
                if diag_idx < len(corner_axes):
                    diag_ax = corner_axes[diag_idx]
                    
                    # Get histogram data from the corner subplot
                    hist_patches = [p for p in diag_ax.patches if hasattr(p, 'get_height')]
                    if hist_patches:
                        # Extract histogram bins and heights
                        bins = []
                        heights = []
                        for patch in hist_patches:
                            bins.append(patch.get_x() + patch.get_width()/2)
                            heights.append(patch.get_height())
                        
                        # Plot the smooth histogram
                        ax.hist(param_samples, bins=40, alpha=0.7, density=True, 
                               color=color, edgecolor='white', linewidth=0.5)
                    
                    # Get line data for smooth curve if available
                    lines = diag_ax.get_lines()
                    for line in lines:
                        if len(line.get_xdata()) > 1:  # Skip single point markers
                            ax.plot(line.get_xdata(), line.get_ydata(), 
                                   color='black', linewidth=1.5, alpha=0.8)
                            break
                
                plt.close(corner_fig)  # Clean up temporary figure
                
            except Exception as e:
                # Fallback to simple histogram + KDE if corner extraction fails
                print(f"Warning: Corner extraction failed for patient {patient_id}, param {param_name}: {e}")
                ax.hist(param_samples, bins=40, alpha=0.7, density=True, 
                       color=color, edgecolor='white', linewidth=0.5)
                
                try:
                    kde = gaussian_kde(param_samples)
                    x_range = np.linspace(param_bounds_j[0], param_bounds_j[1], 200)
                    # Ensure x_range is within sample bounds
                    x_min, x_max = np.min(param_samples), np.max(param_samples)
                    x_range = x_range[(x_range >= x_min) & (x_range <= x_max)]
                    if len(x_range) > 0:
                        kde_values = kde(x_range)
                        ax.plot(x_range, kde_values, color='black', linewidth=1.5, alpha=0.8)
                except Exception:
                    pass  # Skip KDE if it fails
            
            # Set x-axis limits to parameter bounds
            ax.set_xlim(param_bounds_j)
            
            # Add patient ID on the leftmost subplot of each row
            if j == 0:
                ax.set_ylabel(f'Patient {patient_id}', fontsize=10, rotation=0, 
                             ha='right', va='center')
            
            # Remove y-tick labels for cleaner look
            ax.set_yticks([])
            
            # Add parameter label on top row
            if i == 0:
                ax.set_title(param_labels[j], fontsize=12, pad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add mean line
            mean_val = np.mean(param_samples)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=1)
    
    # Add overall title
    fig.suptitle('Parameter Posterior Distributions by Patient', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # Save the plot
    if output_path is None:
        output_path = Path('parameter_posteriors_grid.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Parameter posteriors grid plot saved: {output_path}")
    return output_path


def create_summary_statistics(parameter_data: Dict[str, np.ndarray], 
                            output_dir: Path) -> Path:
    """
    Create summary statistics CSV file for parameter data.
    
    Parameters:
    -----------
    parameter_data : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to parameter sample arrays
    output_dir : Path
        Directory to save the summary file
        
    Returns:
    --------
    Path
        Path to the saved CSV file
    """
    param_names = ['β', 'ρ', 'π', 'φ', 'δ', 'log₁₀V₀']
    summary_data = []
    
    # Create TEIRVInference instance for parameter transformation
    temp_inference = TEIRVInference()
    
    for patient_id, samples in parameter_data.items():
        # Transform samples to match corner plot ordering
        samples_tensor = torch.tensor(samples, dtype=torch.float32)
        transformed_samples = temp_inference._prepare_samples_for_display(samples_tensor)
        transformed_samples_np = transformed_samples.numpy()
        
        for j, param_name in enumerate(param_names):
            param_samples = transformed_samples_np[:, j]
            
            stats = {
                'Patient_ID': patient_id,
                'Parameter': param_name,
                'Mean': np.mean(param_samples),
                'Median': np.median(param_samples),
                'Std': np.std(param_samples),
                'Q25': np.percentile(param_samples, 25),
                'Q75': np.percentile(param_samples, 75),
                'Min': np.min(param_samples),
                'Max': np.max(param_samples),
                'N_Samples': len(param_samples)
            }
            summary_data.append(stats)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary statistics
    summary_path = output_dir / 'parameter_posteriors_summary.csv'
    df_summary.to_csv(summary_path, index=False, float_format='%.4f')
    
    print(f"✅ Parameter posteriors summary saved: {summary_path}")
    return summary_path


def main():
    """Main function to orchestrate parameter posterior grid plotting."""
    parser = argparse.ArgumentParser(
        description='Create parameter posterior grid plots for TEIRV production runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create parameter posteriors grid for all patients
  python create_parameter_posteriors_grid.py 20250704_134546
  
  # Process specific patients only
  python create_parameter_posteriors_grid.py 20250704_134546 --patients 432192 443108
  
  # Save to custom location with custom size
  python create_parameter_posteriors_grid.py 20250704_134546 --output /path/to/plot.png --figsize 20 15
        """
    )
    
    parser.add_argument('run_id', type=str, help='Production run ID (e.g., 20250704_134546)')
    
    # Optional arguments
    parser.add_argument('--output', type=str,
                       help='Custom output path for parameter posteriors grid plot')
    parser.add_argument('--patients', nargs='+', type=str,
                       help='Specific patient IDs to process (default: all)')
    parser.add_argument('--figsize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Custom figure size in inches (e.g., --figsize 20 15)')
    parser.add_argument('--smooth', type=float, default=1.0,
                       help='Smoothing parameter for density estimation (default: 1.0)')
    
    args = parser.parse_args()
    
    print("📊 TEIRV Parameter Posteriors Grid Creator")
    print("=" * 60)
    print(f"Production run: {args.run_id}")
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
        
        # Load parameter data for all patients
        print(f"\\n📊 Loading parameter data for {len(patient_ids)} patients...")
        parameter_data = load_parameter_data(inference_dir, patient_ids)
        
        if not parameter_data:
            print("❌ No parameter data loaded successfully")
            return
        
        # Create parameter posteriors grid plot
        print(f"\\n🎨 Creating parameter posteriors grid plot...")
        output_path = Path(args.output) if args.output else inference_dir / 'parameter_posteriors_grid.png'
        figsize = tuple(args.figsize) if args.figsize else None
        
        plot_path = create_parameter_grid_plot(parameter_data, output_path, figsize, args.smooth)
        
        # Create summary statistics
        summary_path = create_summary_statistics(parameter_data, inference_dir)
        
        print(f"\\n✅ PARAMETER POSTERIORS GRID COMPLETED")
        print(f"📊 Successfully processed {len(parameter_data)}/{len(patient_ids)} patients")
        print(f"📈 Plot saved: {plot_path}")
        print(f"📋 Summary saved: {summary_path}")
        
        # Print brief summary statistics
        print(f"\\n📈 PARAMETER SUMMARY")
        print("=" * 40)
        param_names = ['β', 'ρ', 'π', 'φ', 'δ', 'log₁₀V₀']
        print(f"{'Parameter':<10} {'Patients':<8} {'Total Samples':<12}")
        print("-" * 40)
        
        # Create TEIRVInference instance for transformation
        temp_inference = TEIRVInference()
        
        for j, param_name in enumerate(param_names):
            all_samples = []
            for samples in parameter_data.values():
                samples_tensor = torch.tensor(samples, dtype=torch.float32)
                transformed_samples = temp_inference._prepare_samples_for_display(samples_tensor)
                all_samples.append(transformed_samples.numpy()[:, j])
            
            all_samples_concat = np.concatenate(all_samples)
            total_samples = len(all_samples_concat)
            n_patients = len(parameter_data)
            print(f"{param_name:<10} {n_patients:<8} {total_samples:<12}")
        
        print(f"\\n🎯 Grid dimensions: {len(parameter_data)} patients × {len(param_names)} parameters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()