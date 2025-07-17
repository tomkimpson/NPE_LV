#!/usr/bin/env python3
"""
Create R₀ (basic reproduction number) plots for TEIRV production runs.

This script calculates and visualizes the posterior distribution of R₀ for each patient
from their inferred parameters.

R₀ = (π × β × T(0)) / (δ × c)

Usage:
    python create_r0_plots.py 20250704_134546
    python create_r0_plots.py 20250704_134546 --output /path/to/output.png
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from scipy.stats import gaussian_kde

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.teirv_utils import calculate_r0


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


def load_r0_data_for_patients(inference_dir: Path, patient_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    Load posterior samples and calculate R₀ for all patients.
    
    Parameters:
    -----------
    inference_dir : Path
        Path to inference results directory
    patient_ids : List[str]
        List of patient IDs to process
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping patient IDs to R₀ sample arrays
    """
    r0_data = {}
    
    for patient_id in patient_ids:
        patient_dir = inference_dir / f"patient_{patient_id}"
        samples_path = patient_dir / 'posterior_samples.npy'
        
        if not samples_path.exists():
            print(f"⚠️  Missing posterior samples for patient {patient_id}")
            continue
            
        try:
            # Load posterior samples
            posterior_samples = np.load(samples_path)
            print(f"📊 Loaded {len(posterior_samples)} samples for patient {patient_id}")
            
            # Calculate R₀
            r0_samples = calculate_r0(posterior_samples)
            r0_data[patient_id] = r0_samples
            
            # Print detailed summary statistics
            mean_r0 = np.mean(r0_samples)
            median_r0 = np.median(r0_samples)
            std_r0 = np.std(r0_samples)
            q25, q75 = np.percentile(r0_samples, [25, 75])
            min_r0, max_r0 = np.min(r0_samples), np.max(r0_samples)
            
            print(f"    R₀ Statistics:")
            print(f"      Mean: {mean_r0:.3f}")
            print(f"      Median: {median_r0:.3f}")
            print(f"      Std Dev: {std_r0:.3f}")
            print(f"      IQR: [{q25:.3f}, {q75:.3f}]")
            print(f"      Range: [{min_r0:.3f}, {max_r0:.3f}]")
            
        except Exception as e:
            print(f"❌ Failed to process patient {patient_id}: {e}")
            continue
    
    return r0_data


def create_r0_plot(r0_data: Dict[str, np.ndarray], output_path: Optional[Path] = None, 
                   plot_style: str = 'density') -> Path:
    """
    Create R₀ distribution plot for all patients using one-sided violin plots.
    
    Parameters:
    -----------
    r0_data : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to R₀ sample arrays
    output_path : Optional[Path]
        Custom output path for the plot
    plot_style : str
        Style of plot: 'density', 'violin', or 'half_density'
        
    Returns:
    --------
    Path
        Path to the saved plot
    """
    if not r0_data:
        raise ValueError("No R₀ data available for plotting")
    
    # Sort patient IDs for consistent ordering
    patient_order = sorted(r0_data.keys())
    n_patients = len(patient_order)
    
    # Prepare data for seaborn
    plot_data = []
    for patient_id, r0_samples in r0_data.items():
        for r0_value in r0_samples:
            plot_data.append({'Patient ID': patient_id, 'R₀': r0_value})
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(max(8, n_patients * 1.5), 8))
    
    # Create violin plot with seaborn
    violin_plot = sns.violinplot(
        data=df, 
        x='Patient ID', 
        y='R₀', 
        inner=None,  # We'll add our own markers
        palette='viridis',
        order=patient_order,
        ax=ax
    )
    
    # For 'half_density' style, modify violin plots to be one-sided
    if plot_style == 'half_density':
        for collection in violin_plot.collections:
            # Get all paths in this collection
            paths = collection.get_paths()
            for path in paths:
                vertices = path.vertices.copy()
                # Find the center x position
                x_center = np.mean(vertices[:, 0])
                
                # Set all x-values on the left side of the center to the center
                vertices[:, 0][vertices[:, 0] < x_center] = x_center
                path.vertices = vertices
    
    # Add mean markers for each patient
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_patients))
    for i, patient_id in enumerate(patient_order):
        r0_samples = r0_data[patient_id]
        if len(r0_samples) > 0:
            mean_r0 = np.mean(r0_samples)
            ax.plot(i, mean_r0, 'o', color='white', markersize=8, 
                   markeredgecolor=colors[i], markeredgewidth=2, zorder=10)
    
    # Formatting
    ax.set_xlabel('Patient ID', fontsize=14)
    ax.set_ylabel('Basic Reproduction Number (R₀)', fontsize=14)
    ax.set_title('Posterior Distribution of R₀ by Patient', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 60)
    
    # Rotate x-axis labels if needed
    if n_patients > 6:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = Path('r0_distributions.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ R₀ distribution plot saved: {output_path}")
    return output_path


def create_r0_plot_violin(r0_data: Dict[str, np.ndarray], output_path: Optional[Path] = None) -> Path:
    """
    Create R₀ distribution plot using seaborn violin plots (original implementation).
    
    Parameters:
    -----------
    r0_data : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to R₀ sample arrays
    output_path : Optional[Path]
        Custom output path for the plot
        
    Returns:
    --------
    Path
        Path to the saved plot
    """
    if not r0_data:
        raise ValueError("No R₀ data available for plotting")
    
    # Prepare data for plotting
    plot_data = []
    for patient_id, r0_samples in r0_data.items():
        for r0_value in r0_samples:
            plot_data.append({'Patient ID': patient_id, 'R₀': r0_value})
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Sort patient IDs for consistent ordering
    patient_order = sorted(r0_data.keys())
    
    # Create violin plot
    ax = sns.violinplot(data=df, x='Patient ID', y='R₀', order=patient_order,
                       palette='viridis', inner='box', alpha=0.7)
    
    # Overlay box plot for clearer summary statistics
    sns.boxplot(data=df, x='Patient ID', y='R₀', order=patient_order,
               width=0.3, boxprops=dict(alpha=0.3), ax=ax)
    
    # Add horizontal line at R₀ = 1 (epidemic threshold)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, 
               label='Epidemic threshold (R₀ = 1)')
    
    # Formatting
    plt.xlabel('Patient ID', fontsize=14)
    plt.ylabel('Basic Reproduction Number (R₀)', fontsize=14)
    plt.title('Posterior Distribution of R₀ by Patient (Violin Plot)', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 60)
    plt.legend(loc='upper right', fontsize=12)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        output_path = Path('r0_distributions_violin.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ R₀ violin plot saved: {output_path}")
    return output_path


def create_summary_statistics(r0_data: Dict[str, np.ndarray], output_dir: Path) -> Path:
    """
    Create summary statistics CSV file for R₀ data.
    
    Parameters:
    -----------
    r0_data : Dict[str, np.ndarray]
        Dictionary mapping patient IDs to R₀ sample arrays
    output_dir : Path
        Directory to save the summary file
        
    Returns:
    --------
    Path
        Path to the saved CSV file
    """
    summary_data = []
    
    for patient_id, r0_samples in r0_data.items():
        stats = {
            'Patient_ID': patient_id,
            'Mean_R0': np.mean(r0_samples),
            'Median_R0': np.median(r0_samples),
            'Std_R0': np.std(r0_samples),
            'Q25_R0': np.percentile(r0_samples, 25),
            'Q75_R0': np.percentile(r0_samples, 75),
            'Min_R0': np.min(r0_samples),
            'Max_R0': np.max(r0_samples),
            'N_Samples': len(r0_samples)
        }
        summary_data.append(stats)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save summary statistics
    summary_path = output_dir / 'r0_summary_statistics.csv'
    df_summary.to_csv(summary_path, index=False, float_format='%.4f')
    
    print(f"✅ R₀ summary statistics saved: {summary_path}")
    return summary_path


def main():
    """Main function to orchestrate R₀ calculation and plotting."""
    parser = argparse.ArgumentParser(
        description='Create R₀ distribution plots for TEIRV production runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create R₀ plot with vertical density curves (default)
  python create_r0_plots.py 20250704_134546
  
  # Create half-density plot (JSF style)
  python create_r0_plots.py 20250704_134546 --style half_density
  
  # Create traditional violin plot
  python create_r0_plots.py 20250704_134546 --style violin
  
  # Save to custom location
  python create_r0_plots.py 20250704_134546 --output /path/to/r0_plot.png
        """
    )
    
    parser.add_argument('run_id', type=str, help='Production run ID (e.g., 20250704_134546)')
    
    # Optional arguments
    parser.add_argument('--output', type=str,
                       help='Custom output path for R₀ plot')
    parser.add_argument('--patients', nargs='+', type=str,
                       help='Specific patient IDs to process (default: all)')
    parser.add_argument('--style', type=str, choices=['density', 'half_density', 'violin'], 
                       default='density',
                       help='Plot style: density (full rotated Gaussian), half_density (JSF style), or violin (default: density)')
    
    args = parser.parse_args()
    
    print("🧬 TEIRV R₀ Analysis")
    print("=" * 50)
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
        
        # Load R₀ data for all patients
        print(f"\\n🧮 Calculating R₀ for {len(patient_ids)} patients...")
        r0_data = load_r0_data_for_patients(inference_dir, patient_ids)
        
        if not r0_data:
            print("❌ No R₀ data calculated successfully")
            return
        
        # Create R₀ plot
        print(f"\\n📊 Creating R₀ distribution plot (style: {args.style})...")
        output_path = Path(args.output) if args.output else inference_dir / 'r0_distributions.png'
        
        if args.style == 'violin':
            # Use original seaborn violin plot
            plot_path = create_r0_plot_violin(r0_data, output_path)
        else:
            # Use new density curve plot
            plot_path = create_r0_plot(r0_data, output_path, args.style)
        
        # Create summary statistics
        summary_path = create_summary_statistics(r0_data, inference_dir)
        
        print(f"\\n✅ R₀ ANALYSIS COMPLETED")
        print(f"📊 Successfully processed {len(r0_data)}/{len(patient_ids)} patients")
        print(f"📈 Plot saved: {plot_path}")
        print(f"📋 Summary saved: {summary_path}")
        
        # Print comprehensive summary statistics
        print(f"\\n📈 COMPREHENSIVE R₀ SUMMARY")
        print("=" * 60)
        
        # Individual patient summary
        print("\\n📊 Individual Patient R₀ Statistics:")
        print(f"{'Patient':<10} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 60)
        
        for patient_id in sorted(r0_data.keys()):
            r0_samples = r0_data[patient_id]
            mean_val = np.mean(r0_samples)
            median_val = np.median(r0_samples)
            std_val = np.std(r0_samples)
            min_val = np.min(r0_samples)
            max_val = np.max(r0_samples)
            print(f"{patient_id:<10} {mean_val:<8.3f} {median_val:<8.3f} {std_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f}")
        
        # Overall statistics
        all_r0_values = np.concatenate(list(r0_data.values()))
        overall_mean = np.mean(all_r0_values)
        overall_median = np.median(all_r0_values)
        overall_std = np.std(all_r0_values)
        overall_q25, overall_q75 = np.percentile(all_r0_values, [25, 75])
        epidemic_fraction = np.mean(all_r0_values > 1.0) * 100
        
        print(f"\\n🌍 Overall R₀ Statistics (All Patients):")
        print(f"   Mean: {overall_mean:.3f}")
        print(f"   Median: {overall_median:.3f}")
        print(f"   Std Dev: {overall_std:.3f}")
        print(f"   IQR: [{overall_q25:.3f}, {overall_q75:.3f}]")
        print(f"   Range: [{np.min(all_r0_values):.3f}, {np.max(all_r0_values):.3f}]")
        print(f"   % Samples > 1.0: {epidemic_fraction:.1f}%")
        print(f"   Total samples: {len(all_r0_values):,}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()