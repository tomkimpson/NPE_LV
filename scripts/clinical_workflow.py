#!/usr/bin/env python3
"""
Complete clinical workflow for TEIRV NPE.

End-to-end pipeline:
1. Validate clinical data compatibility
2. Load and preprocess patient data
3. Perform NPE inference on all patients
4. Compare with JSFGermano2024 particle filter results
5. Generate comprehensive clinical report
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from TEIRV.clinical_data import ClinicalStudy, validate_clinical_data_compatibility
from TEIRV.teirv_inference import TEIRVInference
from TEIRV.teirv_utils import create_teirv_time_grid


def validate_prerequisites(model_path: str, data_dir: Optional[str] = None) -> bool:
    """
    Validate that all prerequisites are met for clinical workflow.
    
    Parameters:
    -----------
    model_path : str
        Path to trained NPE model
    data_dir : str, optional
        Path to clinical data directory
        
    Returns:
    --------
    valid : bool
        Whether prerequisites are met
    """
    print("Validating Clinical Workflow Prerequisites")
    print("=" * 50)
    
    # 1. Check NPE model exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ NPE model not found: {model_path}")
        return False
    print(f"✅ NPE model found: {model_path}")
    
    # 2. Validate clinical data
    try:
        if not validate_clinical_data_compatibility():
            print("❌ Clinical data validation failed")
            return False
    except Exception as e:
        print(f"❌ Clinical data validation error: {e}")
        return False
    
    # 3. Check that we can load the NPE model
    try:
        npe = TEIRVInference()
        npe.load_model(model_path)
        print("✅ NPE model loads successfully")
    except Exception as e:
        print(f"❌ Failed to load NPE model: {e}")
        return False
    
    print("\n✅ All prerequisites validated successfully")
    return True


def load_particle_filter_benchmark(data_dir: Optional[str] = None) -> Dict:
    """
    Load particle filter benchmark results from JSFGermano2024.
    
    Parameters:
    -----------
    data_dir : str, optional
        Path to JSFGermano2024 directory
        
    Returns:
    --------
    benchmark_results : dict
        Particle filter parameter estimates for comparison
    """
    print("Loading particle filter benchmark results...")
    
    if data_dir is None:
        # Default location in submodule
        benchmark_dir = Path(__file__).parent.parent / "external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/processing/COVID_Results"
    else:
        benchmark_dir = Path(data_dir)
    
    benchmark_results = {}
    
    # Patient data is in PatientData subdirectory
    patient_data_dir = benchmark_dir / "PatientData"
    if not patient_data_dir.exists():
        print(f"⚠️  Particle filter benchmark data not found at {patient_data_dir}")
        return {}
    
    # Look for parameter estimate files (would need to check actual format)
    patient_files = list(patient_data_dir.glob("*.ssv"))
    print(f"Found {len(patient_files)} patient files in benchmark data")
    
    # For now, return placeholder - would need to implement actual parsing
    # based on the specific format of JSFGermano2024 results
    for patient_file in patient_files:
        patient_id = patient_file.stem
        benchmark_results[patient_id] = {
            'source': 'particle_filter',
            'file': str(patient_file),
            'note': 'Benchmark parsing not yet implemented'
        }
    
    return benchmark_results


def generate_clinical_report(npe_results: pd.DataFrame,
                           benchmark_results: Dict,
                           output_dir: str):
    """
    Generate comprehensive clinical report.
    
    Parameters:
    -----------
    npe_results : pd.DataFrame
        NPE parameter estimates
    benchmark_results : dict
        Particle filter benchmark results
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    
    print("\nGenerating Clinical Report")
    print("=" * 40)
    
    # 1. Summary statistics table
    report_lines = [
        "# TEIRV Clinical Inference Report",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Study Overview",
        f"- **Patients analyzed:** {len(npe_results)}",
        f"- **NPE model:** PyTorch-based Neural Posterior Estimation",
        f"- **Comparison:** JSFGermano2024 particle filter benchmark",
        "",
        "## Parameter Estimates",
        "",
    ]
    
    # Add parameter summary table
    param_cols = [col for col in npe_results.columns if col.endswith('_mean')]
    if param_cols:
        param_summary = npe_results[['patient_id'] + param_cols].round(3)
        report_lines.append("### NPE Parameter Estimates (Mean Values)")
        report_lines.append("")
        report_lines.append(param_summary.to_markdown(index=False))
        report_lines.append("")
    
    # Add credible interval information
    ci_cols = [col for col in npe_results.columns if col.endswith('_q025') or col.endswith('_q975')]
    if ci_cols:
        report_lines.append("### Parameter Credible Intervals (95%)")
        report_lines.append("")
        
        # Reorganize CI data for better presentation
        for patient_id in npe_results['patient_id']:
            patient_data = npe_results[npe_results['patient_id'] == patient_id].iloc[0]
            report_lines.append(f"**Patient {patient_id}:**")
            
            param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
            for param in param_names:
                mean_col = f'{param}_mean'
                q025_col = f'{param}_q025'
                q975_col = f'{param}_q975'
                
                if all(col in patient_data.index for col in [mean_col, q025_col, q975_col]):
                    mean_val = patient_data[mean_col]
                    ci_lower = patient_data[q025_col]
                    ci_upper = patient_data[q975_col]
                    report_lines.append(f"- {param}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            report_lines.append("")
    
    # Add benchmark comparison if available
    if benchmark_results:
        report_lines.append("## Comparison with Particle Filter Benchmark")
        report_lines.append("")
        report_lines.append("### Benchmark Data Available")
        for patient_id, benchmark in benchmark_results.items():
            report_lines.append(f"- Patient {patient_id}: {benchmark.get('note', 'Available')}")
        report_lines.append("")
        report_lines.append("*Note: Detailed comparison requires parsing benchmark parameter estimates*")
        report_lines.append("")
    
    # Add methodology section
    report_lines.extend([
        "## Methodology",
        "",
        "### TEIRV Model",
        "- **Compartments:** T (target), E (eclipsed), I (infectious), R (refractory), V (virions)",
        "- **Reactions:** 7 stochastic reactions modeling viral dynamics",
        "- **Parameters:** 6 inferred [β, π, δ, φ, ρ, V₀] + 2 fixed [k=4, c=10]",
        "",
        "### NPE Implementation",
        "- **Simulator:** Gillespie algorithm with PyTorch implementation",
        "- **Neural Network:** Normalizing flows with coupling transforms",
        "- **Training Data:** Synthetic RT-PCR observations with observation noise",
        "- **Priors:** Uniform distributions matching JSFGermano2024 paper",
        "",
        "### Clinical Data",
        "- **Source:** JSFGermano2024 COVID-19 patient RT-PCR data",
        "- **Observations:** Log₁₀ viral load with detection limit -0.65",
        "- **Time Grid:** Daily measurements over ~14 days",
        "",
        "## Technical Details",
        f"- **Posterior Samples:** {npe_results['n_samples'].iloc[0] if 'n_samples' in npe_results.columns else 'N/A'} per patient",
        f"- **Mean Inference Time:** {npe_results['inference_time'].mean():.2f}s per patient" if 'inference_time' in npe_results.columns else "",
        "- **Implementation:** Phase 3 TEIRV NPE pipeline",
        "",
        "---",
        "",
        "*Generated by TEIRV NPE Clinical Workflow*"
    ])
    
    # Save report
    report_path = output_path / 'clinical_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Clinical report saved to {report_path}")
    
    # Also create a summary CSV for easy analysis
    summary_path = output_path / 'clinical_summary.csv'
    npe_results.to_csv(summary_path, index=False)
    print(f"✅ Summary data saved to {summary_path}")


def create_comparison_plots(npe_results: pd.DataFrame,
                          benchmark_results: Dict,
                          output_dir: str):
    """
    Create comparison plots between NPE and particle filter results.
    
    Parameters:
    -----------
    npe_results : pd.DataFrame
        NPE parameter estimates
    benchmark_results : dict
        Particle filter benchmark results  
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    
    print("Creating comparison plots...")
    
    # 1. Parameter distribution plots
    param_names = ['β', 'π', 'δ', 'φ', 'ρ', 'V₀']
    param_cols = [f'{param}_mean' for param in param_names]
    
    available_params = [col for col in param_cols if col in npe_results.columns]
    
    if available_params:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param_col in enumerate(available_params):
            if i >= len(axes):
                break
                
            ax = axes[i]
            param_name = param_col.replace('_mean', '')
            
            # Plot NPE estimates
            values = npe_results[param_col]
            patients = npe_results['patient_id']
            
            ax.scatter(range(len(values)), values, alpha=0.7, s=100, label='NPE')
            
            # Add error bars if credible intervals available
            ci_lower_col = f'{param_name}_q025'
            ci_upper_col = f'{param_name}_q975'
            
            if ci_lower_col in npe_results.columns and ci_upper_col in npe_results.columns:
                lower = npe_results[ci_lower_col]
                upper = npe_results[ci_upper_col]
                ax.errorbar(range(len(values)), values, 
                           yerr=[values - lower, upper - values],
                           fmt='none', alpha=0.5)
            
            ax.set_xlabel('Patient')
            ax.set_ylabel(f'{param_name}')
            ax.set_title(f'Parameter {param_name}')
            ax.set_xticks(range(len(patients)))
            ax.set_xticklabels([p[:6] for p in patients], rotation=45)
            
            # TODO: Add particle filter results when benchmark parsing is implemented
            
        # Hide unused subplots
        for i in range(len(available_params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Parameter comparison plot saved")
    
    # 2. Clinical correlation analysis
    if len(npe_results) > 1:
        # Create correlation matrix of parameter estimates
        param_data = npe_results[[col for col in param_cols if col in npe_results.columns]]
        
        if len(param_data.columns) > 1:
            corr_matrix = param_data.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'label': 'Correlation'})
            plt.title('Parameter Correlation Matrix Across Patients')
            plt.tight_layout()
            plt.savefig(output_path / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Parameter correlation plot saved")


def main():
    parser = argparse.ArgumentParser(description='Complete TEIRV clinical workflow')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained NPE model (.pkl file)')
    parser.add_argument('--output', type=str, default='results/clinical_workflow',
                       help='Output directory for results')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of posterior samples per patient')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to clinical data directory')
    parser.add_argument('--benchmark_dir', type=str, default=None,
                       help='Path to particle filter benchmark results')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip prerequisite validation')
    
    args = parser.parse_args()
    
    print("TEIRV Clinical Workflow Pipeline")
    print("=" * 50)
    
    # 1. Validate prerequisites
    if not args.skip_validation:
        if not validate_prerequisites(args.model, args.data_dir):
            print("❌ Prerequisite validation failed")
            return
    
    # 2. Load benchmark data
    benchmark_results = load_particle_filter_benchmark(args.benchmark_dir)
    
    # 3. Run clinical inference
    print("\nRunning clinical inference...")
    
    # Import and call the clinical inference script
    from fit_clinical_data import main as run_clinical_inference
    
    # Temporarily modify sys.argv to pass arguments
    original_argv = sys.argv.copy()
    sys.argv = [
        'fit_clinical_data.py',
        '--model', args.model,
        '--output', args.output,
        '--n_samples', str(args.n_samples)
    ]
    
    if args.data_dir:
        sys.argv.extend(['--data_dir', args.data_dir])
    
    try:
        run_clinical_inference()
        print("✅ Clinical inference completed")
    except Exception as e:
        print(f"❌ Clinical inference failed: {e}")
        return
    finally:
        sys.argv = original_argv
    
    # 4. Load inference results
    results_path = Path(args.output)
    summary_file = results_path / 'clinical_parameter_estimates.csv'
    
    if not summary_file.exists():
        print(f"❌ Results file not found: {summary_file}")
        return
    
    npe_results = pd.read_csv(summary_file)
    print(f"✅ Loaded results for {len(npe_results)} patients")
    
    # 5. Generate comprehensive report
    generate_clinical_report(npe_results, benchmark_results, args.output)
    
    # 6. Create comparison plots
    create_comparison_plots(npe_results, benchmark_results, args.output)
    
    print(f"\n✅ Clinical workflow completed successfully")
    print(f"Results available in: {args.output}")
    print(f"  - clinical_report.md: Comprehensive clinical report")
    print(f"  - clinical_parameter_estimates.csv: Parameter estimates")
    print(f"  - parameter_comparison.png: Visualization of results")
    print(f"  - patient_*/: Individual patient results")


if __name__ == '__main__':
    main()