#!/usr/bin/env python3
"""
JSF TEIRV Examples

This script demonstrates various ways to use the deploy_jsf_teirv.py script
for running TEIRV simulations with the new JSF framework.

Examples include:
- Basic single simulation
- Parameter exploration
- Method comparisons
- Batch processing
- Custom configurations

Author: Generated for NPE_LV project
Date: 2025-01-17
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add the scripts directory to path
sys.path.append(str(Path(__file__).parent))

from deploy_jsf_teirv import (
    TEIRVDeployment,
    TEIRVParameters,
    TEIRVInitialConditions,
    TEIRVSimulationConfig,
    create_default_parameters,
    create_default_initial_conditions,
    create_default_config
)


def example_basic_simulation():
    """Example 1: Basic single simulation with default parameters."""
    print("="*60)
    print("Example 1: Basic Single Simulation")
    print("="*60)
    
    # Create deployment manager
    deployment = TEIRVDeployment("examples/basic_simulation")
    
    # Use default parameters and configuration
    parameters = create_default_parameters()
    initial_conditions = create_default_initial_conditions()
    config = create_default_config()
    
    # Run simulation
    result = deployment.run_single_simulation(
        parameters=parameters,
        initial_conditions=initial_conditions,
        config=config,
        name="basic_teirv"
    )
    
    # Save results and create plots
    deployment.save_results("basic_results.json")
    deployment.create_summary_plots()
    
    print(f"✅ Basic simulation completed")
    print(f"📊 Generated {len(result['times'])} time points")
    print(f"⏱️  Simulation time: {result['simulation_time']:.3f} seconds")
    print(f"📁 Results saved to: {deployment.output_dir}")
    
    return result


def example_parameter_exploration():
    """Example 2: Explore the effect of different infection rates (beta)."""
    print("\n" + "="*60)
    print("Example 2: Parameter Exploration (Beta values)")
    print("="*60)
    
    # Create deployment manager
    deployment = TEIRVDeployment("examples/parameter_exploration")
    
    # Base parameters and configuration
    base_parameters = create_default_parameters()
    initial_conditions = create_default_initial_conditions()
    config = create_default_config()
    
    # Define parameter ranges for beta
    beta_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    parameter_ranges = {'beta': beta_values}
    
    # Run parameter sweep
    sweep_results = deployment.run_parameter_sweep(
        base_parameters=base_parameters,
        base_initial_conditions=initial_conditions,
        base_config=config,
        parameter_ranges=parameter_ranges,
        name_prefix="beta_sweep"
    )
    
    # Save results and create plots
    deployment.save_results("parameter_sweep_results.json")
    deployment.create_summary_plots()
    
    print(f"✅ Parameter exploration completed")
    print(f"📊 {len(sweep_results)} simulations run")
    print(f"📁 Results saved to: {deployment.output_dir}")
    
    return sweep_results


def example_method_comparison():
    """Example 3: Compare exact vs operator-splitting methods."""
    print("\n" + "="*60)
    print("Example 3: Method Comparison")
    print("="*60)
    
    # Create deployment manager
    deployment = TEIRVDeployment("examples/method_comparison")
    
    # Use default parameters and initial conditions
    parameters = create_default_parameters()
    initial_conditions = create_default_initial_conditions()
    
    # Run with exact method
    config_exact = TEIRVSimulationConfig(
        t_max=10.0,
        method='exact',
        switching_threshold=100.0
    )
    
    result_exact = deployment.run_single_simulation(
        parameters=parameters,
        initial_conditions=initial_conditions,
        config=config_exact,
        name="exact_method"
    )
    
    # Run with operator-splitting method
    config_op_split = TEIRVSimulationConfig(
        t_max=10.0,
        dt=0.00005,
        method='operator-splitting',
        switching_threshold=100.0
    )
    
    result_op_split = deployment.run_single_simulation(
        parameters=parameters,
        initial_conditions=initial_conditions,
        config=config_op_split,
        name="operator_splitting"
    )
    
    # Save results and create comparison plots
    deployment.save_results("method_comparison_results.json")
    deployment.create_summary_plots()
    
    # Create detailed comparison plot
    create_method_comparison_plot(result_exact, result_op_split, deployment.output_dir)
    
    print(f"✅ Method comparison completed")
    print(f"⏱️  Exact method time: {result_exact['simulation_time']:.3f} seconds")
    print(f"⏱️  Operator-splitting time: {result_op_split['simulation_time']:.3f} seconds")
    print(f"📁 Results saved to: {deployment.output_dir}")
    
    return result_exact, result_op_split


def example_custom_scenario():
    """Example 4: Custom scenario with high initial viral load."""
    print("\n" + "="*60)
    print("Example 4: Custom High Viral Load Scenario")
    print("="*60)
    
    # Create deployment manager
    deployment = TEIRVDeployment("examples/custom_scenario")
    
    # Custom parameters: higher infection rate and virus production
    parameters = TEIRVParameters(
        beta=2.0,           # Higher infection rate
        phi=1.0,            # Standard interferon protection
        rho=0.05,           # Lower reversion rate
        k=4.0,              # Standard eclipse rate
        delta=1.5,          # Lower infected cell death rate
        pi=600.0,           # Higher virus production
        c=8.0               # Lower virus clearance
    )
    
    # Custom initial conditions: high initial viral load
    initial_conditions = TEIRVInitialConditions(
        T=8e7,              # Standard target cells
        E=0.0,              # No eclipse cells initially
        I=0.0,              # No infected cells initially
        R=0.0,              # No refractory cells initially
        V=1000.0            # High initial viral load
    )
    
    # Extended simulation time to see full dynamics
    config = TEIRVSimulationConfig(
        t_max=15.0,         # Longer simulation
        method='exact',
        switching_threshold=100.0
    )
    
    # Run simulation
    result = deployment.run_single_simulation(
        parameters=parameters,
        initial_conditions=initial_conditions,
        config=config,
        name="high_viral_load"
    )
    
    # Save results and create plots
    deployment.save_results("custom_scenario_results.json")
    deployment.create_summary_plots()
    
    print(f"✅ Custom scenario completed")
    print(f"🦠 Initial viral load: {initial_conditions.V}")
    print(f"📊 Final viral load: {result['trajectories'][-1, 4]:.2e}")
    print(f"📁 Results saved to: {deployment.output_dir}")
    
    return result


def example_multi_parameter_sweep():
    """Example 5: Multi-parameter sweep (beta and pi)."""
    print("\n" + "="*60)
    print("Example 5: Multi-Parameter Sweep (Beta and Pi)")
    print("="*60)
    
    # Create deployment manager
    deployment = TEIRVDeployment("examples/multi_parameter_sweep")
    
    # Base parameters and configuration
    base_parameters = create_default_parameters()
    initial_conditions = create_default_initial_conditions()
    config = create_default_config()
    
    # Define parameter ranges
    parameter_ranges = {
        'beta': [0.5, 1.0, 2.0],      # Infection rate
        'pi': [200.0, 400.0, 800.0]   # Virus production rate
    }
    
    # Run parameter sweep
    sweep_results = deployment.run_parameter_sweep(
        base_parameters=base_parameters,
        base_initial_conditions=initial_conditions,
        base_config=config,
        parameter_ranges=parameter_ranges,
        name_prefix="multi_sweep"
    )
    
    # Save results and create plots
    deployment.save_results("multi_parameter_sweep_results.json")
    deployment.create_summary_plots()
    
    # Create heatmap of final viral loads
    create_parameter_heatmap(sweep_results, parameter_ranges, deployment.output_dir)
    
    print(f"✅ Multi-parameter sweep completed")
    print(f"📊 {len(sweep_results)} simulations run")
    print(f"📁 Results saved to: {deployment.output_dir}")
    
    return sweep_results


def create_method_comparison_plot(result_exact, result_op_split, output_dir):
    """Create detailed comparison plot for different methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    compartments = ['T', 'E', 'I', 'R', 'V']
    colors = ['blue', 'red']
    methods = ['Exact', 'Operator-Splitting']
    results = [result_exact, result_op_split]
    
    for i, compartment in enumerate(compartments):
        ax = axes[i]
        
        for j, (result, color, method) in enumerate(zip(results, colors, methods)):
            times = result['times']
            trajectories = result['trajectories']
            
            ax.plot(times, trajectories[:, i], color=color, label=method, 
                   linewidth=2 if j == 0 else 1, alpha=0.8)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(f'{compartment} count')
        ax.set_title(f'Compartment {compartment}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Use log scale for large values
        if compartment in ['T', 'V']:
            ax.set_yscale('log')
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Method comparison plot saved to {output_dir / 'method_comparison_detailed.png'}")


def create_parameter_heatmap(sweep_results, parameter_ranges, output_dir):
    """Create heatmap showing final viral loads for parameter combinations."""
    # Extract parameter values
    beta_values = parameter_ranges['beta']
    pi_values = parameter_ranges['pi']
    
    # Create result matrix
    result_matrix = np.zeros((len(beta_values), len(pi_values)))
    
    for name, result in sweep_results.items():
        # Extract parameter values from result
        beta = result['parameters']['beta']
        pi = result['parameters']['pi']
        
        # Find indices
        beta_idx = beta_values.index(beta)
        pi_idx = pi_values.index(pi)
        
        # Get final viral load
        final_viral_load = result['trajectories'][-1, 4]  # V compartment
        result_matrix[beta_idx, pi_idx] = final_viral_load
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(result_matrix, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(pi_values)))
    ax.set_yticks(range(len(beta_values)))
    ax.set_xticklabels([f'{pi:.0f}' for pi in pi_values])
    ax.set_yticklabels([f'{beta:.1f}' for beta in beta_values])
    
    # Add labels
    ax.set_xlabel('Pi (virus production rate)')
    ax.set_ylabel('Beta (infection rate)')
    ax.set_title('Final Viral Load Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Viral Load')
    
    # Add text annotations
    for i in range(len(beta_values)):
        for j in range(len(pi_values)):
            text = ax.text(j, i, f'{result_matrix[i, j]:.1e}',
                         ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Parameter heatmap saved to {output_dir / 'parameter_heatmap.png'}")


def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Running All JSF TEIRV Examples")
    print("="*60)
    
    start_time = time.time()
    
    # Run examples
    examples = [
        ("Basic Simulation", example_basic_simulation),
        ("Parameter Exploration", example_parameter_exploration),
        ("Method Comparison", example_method_comparison),
        ("Custom Scenario", example_custom_scenario),
        ("Multi-Parameter Sweep", example_multi_parameter_sweep)
    ]
    
    results = {}
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running {name}...")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 ALL EXAMPLES COMPLETED")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ {len(results)} examples successful")
    print(f"📁 Results saved in examples/ directory")
    
    return results


if __name__ == '__main__':
    """Run examples based on command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='JSF TEIRV Examples')
    parser.add_argument('example', nargs='?', default='all',
                       choices=['all', 'basic', 'parameter', 'method', 'custom', 'multi'],
                       help='Which example to run')
    
    args = parser.parse_args()
    
    if args.example == 'all':
        run_all_examples()
    elif args.example == 'basic':
        example_basic_simulation()
    elif args.example == 'parameter':
        example_parameter_exploration()
    elif args.example == 'method':
        example_method_comparison()
    elif args.example == 'custom':
        example_custom_scenario()
    elif args.example == 'multi':
        example_multi_parameter_sweep()
    else:
        print(f"Unknown example: {args.example}")