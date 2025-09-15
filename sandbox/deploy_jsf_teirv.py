#!/usr/bin/env python3
"""
JSF TEIRV Deployment Script

This script provides a modern interface for deploying the TEIRV (Target-Eclipse-Infected-Refractory-Virus) 
epidemiological model using the updated JSF (Jump-Switch-Flow) framework.

The script bridges the old JSF TEIRV implementation with the new JSF framework, providing:
- Easy parameter configuration
- Flexible initial conditions
- Choice of simulation methods (exact vs operator-splitting)
- Structured output management
- Batch processing capabilities

Author: T. Kimpson
Date: 2025-01-17
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import time
import json

# Add external JSF to path
sys.path.append(str(Path(__file__).parent.parent / 'external' / 'jsf'))

try:
    import jsf
    from jsf.types import SystemState, Time, Trajectory
except ImportError as e:
    print(f"Error importing JSF framework: {e}")
    print("Please ensure the JSF framework is properly installed in external/jsf/")
    sys.exit(1)


@dataclass
class TEIRVParameters:
    """TEIRV model parameters with validation."""
    
    # Core epidemiological parameters
    beta: float = 1e-9      # Infection rate (1/virion/day)
    phi: float = 1e-5       # Interferon protection rate (1/day)
    rho: float = 0.1        # Reversion rate from R to T (1/day)
    k: float = 4.0          # Eclipse to infected transition rate (1/day)
    delta: float = 2.0      # Infected cell death rate (1/day)
    pi: float = 400.0       # Virus production rate (virions/infected_cell/day)
    c: float = 10.0         # Virus clearance rate (1/day)
    
    def __post_init__(self):
        """Validate parameter values."""
        self.validate()
    
    def validate(self):
        """Validate that all parameters are within reasonable bounds."""
        validations = [
            (self.beta > 0, "beta must be positive"),
            (self.phi > 0, "phi must be positive"),
            (self.rho >= 0, "rho must be non-negative"),
            (self.k > 0, "k must be positive"),
            (self.delta > 0, "delta must be positive"),
            (self.pi > 0, "pi must be positive"),
            (self.c > 0, "c must be positive"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Parameter validation failed: {message}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy serialization."""
        return {
            'beta': self.beta,
            'phi': self.phi,
            'rho': self.rho,
            'k': self.k,
            'delta': self.delta,
            'pi': self.pi,
            'c': self.c
        }
    
    @classmethod
    def from_dict(cls, param_dict: Dict[str, float]) -> 'TEIRVParameters':
        """Create from dictionary."""
        return cls(**param_dict)


@dataclass
class TEIRVInitialConditions:
    """TEIRV initial conditions with validation."""
    
    T: float = 8e7          # Target cells
    E: float = 0.0          # Eclipse cells
    I: float = 0.0          # Infected cells
    R: float = 0.0          # Refractory cells
    V: float = 1.0          # Virions
    
    def __post_init__(self):
        """Validate initial conditions."""
        self.validate()
    
    def validate(self):
        """Validate that all initial conditions are non-negative."""
        validations = [
            (self.T >= 0, "T must be non-negative"),
            (self.E >= 0, "E must be non-negative"),
            (self.I >= 0, "I must be non-negative"),
            (self.R >= 0, "R must be non-negative"),
            (self.V >= 0, "V must be non-negative"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Initial condition validation failed: {message}")
    
    def to_list(self) -> List[float]:
        """Convert to list for JSF simulation."""
        return [self.T, self.E, self.I, self.R, self.V]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'T': self.T,
            'E': self.E,
            'I': self.I,
            'R': self.R,
            'V': self.V
        }


@dataclass
class TEIRVSimulationConfig:
    """Configuration for TEIRV simulation."""
    
    t_max: float = 10.0                    # Maximum simulation time (days)
    dt: float = 0.00005                    # Time step for operator splitting
    method: str = 'exact'                  # Simulation method: 'exact' or 'operator-splitting'
    switching_threshold: float = 100.0     # Threshold for discrete/continuous switching
    output_times: Optional[List[float]] = None  # Specific output times (if None, use default grid)
    
    def __post_init__(self):
        """Validate configuration."""
        self.validate()
        
        # Set default output times if not provided
        if self.output_times is None:
            self.output_times = np.linspace(0, self.t_max, int(self.t_max) + 1).tolist()
    
    def validate(self):
        """Validate simulation configuration."""
        validations = [
            (self.t_max > 0, "t_max must be positive"),
            (self.dt > 0, "dt must be positive"),
            (self.method in ['exact', 'operator-splitting'], "method must be 'exact' or 'operator-splitting'"),
            (self.switching_threshold > 0, "switching_threshold must be positive"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Configuration validation failed: {message}")


class TEIRVJSFModel:
    """TEIRV model implementation using the JSF framework."""
    
    def __init__(self, parameters: TEIRVParameters, initial_conditions: TEIRVInitialConditions):
        """
        Initialize TEIRV model.
        
        Args:
            parameters: TEIRV model parameters
            initial_conditions: Initial conditions for all compartments
        """
        self.parameters = parameters
        self.initial_conditions = initial_conditions
        self._setup_stoichiometry()
    
    def _setup_stoichiometry(self):
        """Set up stoichiometry matrices based on TEIRV reactions."""
        # TEIRV Reactions:
        # 1. T + V -> E + V     (infection)
        # 2. T + I -> R + I     (interferon protection)
        # 3. R -> T             (reversion)
        # 4. E -> I             (eclipse to infected)
        # 5. I -> 0             (infected cell death)
        # 6. I -> I + V         (virus production)
        # 7. V -> 0             (virus clearance)
        
        # Reactants matrix: [T, E, I, R, V]
        self.nu_reactants = [
            [1, 0, 0, 0, 1],  # T + V -> E + V
            [1, 0, 1, 0, 0],  # T + I -> R + I
            [0, 0, 0, 1, 0],  # R -> T
            [0, 1, 0, 0, 0],  # E -> I
            [0, 0, 1, 0, 0],  # I -> 0
            [0, 0, 1, 0, 0],  # I -> I + V
            [0, 0, 0, 0, 1],  # V -> 0
        ]
        
        # Products matrix: [T, E, I, R, V]
        self.nu_products = [
            [0, 1, 0, 0, 1],  # T + V -> E + V
            [0, 0, 1, 1, 0],  # T + I -> R + I
            [1, 0, 0, 0, 0],  # R -> T
            [0, 0, 1, 0, 0],  # E -> I
            [0, 0, 0, 0, 0],  # I -> 0
            [0, 0, 1, 0, 1],  # I -> I + V
            [0, 0, 0, 0, 0],  # V -> 0
        ]
        
        # Net stoichiometry matrix
        self.nu = [[p - r for p, r in zip(prod, react)] 
                   for prod, react in zip(self.nu_products, self.nu_reactants)]
        
        # Complete stoichiometry dictionary for JSF
        self.stoich = {
            'nu': self.nu,
            'DoDisc': [0, 0, 0, 0, 0],  # All compartments can be continuous
            'nuReactant': self.nu_reactants,
            'nuProduct': self.nu_products
        }
    
    def rates(self, x: List[float], t: float) -> List[float]:
        """
        Compute reaction rates for current state.
        
        Args:
            x: Current state [T, E, I, R, V]
            t: Current time
            
        Returns:
            List of reaction rates
        """
        T, E, I, R, V = x
        
        # Convert parameters to appropriate scales (matching old implementation)
        beta = self.parameters.beta * 1e-9  # Scale factor from old implementation
        phi = self.parameters.phi * 1e-5    # Scale factor from old implementation
        rho = self.parameters.rho
        k = self.parameters.k
        delta = self.parameters.delta
        pi = self.parameters.pi
        c = self.parameters.c
        
        # Reaction rates (matching old implementation)
        rates = [
            beta * T * V,      # Infection: T + V -> E + V
            phi * I * T,       # Interferon protection: T + I -> R + I
            rho * R,           # Reversion: R -> T
            k * E,             # Eclipse to infected: E -> I
            delta * I,         # Infected cell death: I -> 0
            pi * I,            # Virus production: I -> I + V
            c * V              # Virus clearance: V -> 0
        ]
        
        return rates
    
    def simulate(self, config: TEIRVSimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run TEIRV simulation using JSF framework.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Tuple of (times, trajectories) where trajectories is shape (n_times, n_compartments)
        """
        # Prepare initial state
        x0 = SystemState(self.initial_conditions.to_list())
        
        # Prepare JSF options
        jsf_options = {
            'dt': config.dt,
            'SwitchingThreshold': [config.switching_threshold] * 5,
            'EnforceDo': [0] * 5  # Allow switching for all compartments
        }
        
        # Run simulation
        try:
            if config.method == 'exact':
                trajectory = jsf.jsf(
                    x0=x0,
                    rates=self.rates,
                    stoich=self.stoich,
                    t_max=config.t_max,
                    method='exact',
                    config=jsf_options
                )
            else:  # operator-splitting
                trajectory = jsf.JumpSwitchFlowSimulator(
                    x0=x0,
                    rates=self.rates,
                    stoich=self.stoich,
                    t_max=config.t_max,
                    options=jsf_options
                )
            
            # Extract results
            compartment_histories = trajectory[0]
            times = np.array(trajectory[1])
            
            # Convert to numpy array format
            trajectories = np.column_stack(compartment_histories)
            
            return times, trajectories
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")


class TEIRVDeployment:
    """Main deployment class for TEIRV JSF simulations."""
    
    def __init__(self, output_dir: str = "teirv_jsf_results"):
        """
        Initialize deployment manager.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        
    def run_single_simulation(self, 
                            parameters: TEIRVParameters, 
                            initial_conditions: TEIRVInitialConditions,
                            config: TEIRVSimulationConfig,
                            name: str = "simulation") -> Dict[str, Any]:
        """
        Run a single TEIRV simulation.
        
        Args:
            parameters: Model parameters
            initial_conditions: Initial conditions
            config: Simulation configuration
            name: Name for this simulation
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running simulation: {name}")
        
        # Create model
        model = TEIRVJSFModel(parameters, initial_conditions)
        
        # Run simulation
        start_time = time.time()
        times, trajectories = model.simulate(config)
        simulation_time = time.time() - start_time
        
        # Package results
        result = {
            'name': name,
            'parameters': parameters.to_dict(),
            'initial_conditions': initial_conditions.to_dict(),
            'config': config.__dict__,
            'times': times,
            'trajectories': trajectories,
            'compartment_names': ['T', 'E', 'I', 'R', 'V'],
            'simulation_time': simulation_time,
            'method': config.method
        }
        
        # Store results
        self.results[name] = result
        
        print(f"  ✅ Completed in {simulation_time:.3f} seconds")
        print(f"  📊 Generated {len(times)} time points")
        
        return result
    
    def run_parameter_sweep(self, 
                          base_parameters: TEIRVParameters,
                          base_initial_conditions: TEIRVInitialConditions,
                          base_config: TEIRVSimulationConfig,
                          parameter_ranges: Dict[str, List[float]],
                          name_prefix: str = "sweep") -> Dict[str, Any]:
        """
        Run parameter sweep across specified parameter ranges.
        
        Args:
            base_parameters: Base parameter set
            base_initial_conditions: Base initial conditions
            base_config: Base simulation configuration
            parameter_ranges: Dictionary of parameter names to lists of values
            name_prefix: Prefix for simulation names
            
        Returns:
            Dictionary with all sweep results
        """
        print(f"Running parameter sweep: {len(parameter_ranges)} parameters")
        
        sweep_results = {}
        total_runs = np.prod([len(values) for values in parameter_ranges.values()])
        
        print(f"  Total simulations: {total_runs}")
        
        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        import itertools
        for i, combination in enumerate(itertools.product(*param_values)):
            # Create modified parameters
            modified_params = base_parameters.to_dict()
            param_string = []
            
            for param_name, value in zip(param_names, combination):
                modified_params[param_name] = value
                param_string.append(f"{param_name}={value}")
            
            # Create simulation name
            sim_name = f"{name_prefix}_{i:03d}_{'_'.join(param_string)}"
            
            # Run simulation
            try:
                result = self.run_single_simulation(
                    parameters=TEIRVParameters.from_dict(modified_params),
                    initial_conditions=base_initial_conditions,
                    config=base_config,
                    name=sim_name
                )
                sweep_results[sim_name] = result
                
            except Exception as e:
                print(f"  ❌ Failed simulation {sim_name}: {e}")
                continue
        
        print(f"✅ Parameter sweep completed: {len(sweep_results)}/{total_runs} successful")
        
        return sweep_results
    
    def save_results(self, filename: str = "teirv_results.json"):
        """Save all results to JSON file."""
        output_file = self.output_dir / filename
        
        # Prepare results for JSON serialization
        json_results = {}
        for name, result in self.results.items():
            json_result = result.copy()
            # Convert numpy arrays to lists
            json_result['times'] = result['times'].tolist()
            json_result['trajectories'] = result['trajectories'].tolist()
            json_results[name] = json_result
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to {output_file}")
    
    def create_summary_plots(self, results_to_plot: Optional[List[str]] = None):
        """Create summary plots for simulation results."""
        if results_to_plot is None:
            results_to_plot = list(self.results.keys())
        
        compartments = ['T', 'E', 'I', 'R', 'V']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, compartment in enumerate(compartments):
            ax = axes[i]
            
            for name in results_to_plot:
                if name in self.results:
                    result = self.results[name]
                    times = result['times']
                    trajectories = result['trajectories']
                    
                    ax.plot(times, trajectories[:, i], label=name, alpha=0.7)
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(f'{compartment} count')
            ax.set_title(f'Compartment {compartment}')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'teirv_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary plots saved to {self.output_dir / 'teirv_summary.png'}")


def create_default_parameters() -> TEIRVParameters:
    """Create default TEIRV parameters based on literature values."""
    return TEIRVParameters(
        beta=1.0,           # Will be scaled by 1e-9 in rates function
        phi=1.0,            # Will be scaled by 1e-5 in rates function
        rho=0.1,            # 1/day
        k=4.0,              # 1/day
        delta=2.0,          # 1/day
        pi=400.0,           # virions/infected_cell/day
        c=10.0              # 1/day
    )


def create_default_initial_conditions() -> TEIRVInitialConditions:
    """Create default initial conditions."""
    return TEIRVInitialConditions(
        T=8e7,              # Target cells
        E=0.0,              # Eclipse cells
        I=0.0,              # Infected cells
        R=0.0,              # Refractory cells
        V=1.0               # Virions
    )


def create_default_config() -> TEIRVSimulationConfig:
    """Create default simulation configuration."""
    return TEIRVSimulationConfig(
        t_max=10.0,
        dt=0.00005,
        method='exact',
        switching_threshold=100.0
    )


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='JSF TEIRV Deployment Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single simulation with default parameters
  python deploy_jsf_teirv.py single
  
  # Run with custom parameters
  python deploy_jsf_teirv.py single --beta 2.0 --pi 500.0 --V0 10.0
  
  # Run parameter sweep
  python deploy_jsf_teirv.py sweep --beta 1.0,2.0,5.0 --pi 200.0,400.0,800.0
  
  # Use operator-splitting method
  python deploy_jsf_teirv.py single --method operator-splitting
        """
    )
    
    # Global arguments
    parser.add_argument('--output_dir', type=str, default='teirv_jsf_results',
                       help='Output directory for results')
    parser.add_argument('--method', type=str, default='exact',
                       choices=['exact', 'operator-splitting'],
                       help='Simulation method')
    parser.add_argument('--t_max', type=float, default=10.0,
                       help='Maximum simulation time (days)')
    parser.add_argument('--dt', type=float, default=0.00005,
                       help='Time step for operator splitting')
    parser.add_argument('--threshold', type=float, default=100.0,
                       help='Switching threshold')
    
    # Parameter arguments
    parser.add_argument('--beta', type=str, default='1.0',
                       help='Beta parameter (infection rate)')
    parser.add_argument('--phi', type=str, default='1.0',
                       help='Phi parameter (interferon protection)')
    parser.add_argument('--rho', type=str, default='0.1',
                       help='Rho parameter (reversion rate)')
    parser.add_argument('--k', type=str, default='4.0',
                       help='K parameter (eclipse to infected rate)')
    parser.add_argument('--delta', type=str, default='2.0',
                       help='Delta parameter (infected cell death rate)')
    parser.add_argument('--pi', type=str, default='400.0',
                       help='Pi parameter (virus production rate)')
    parser.add_argument('--c', type=str, default='10.0',
                       help='C parameter (virus clearance rate)')
    
    # Initial condition arguments
    parser.add_argument('--T0', type=float, default=8e7,
                       help='Initial target cells')
    parser.add_argument('--E0', type=float, default=0.0,
                       help='Initial eclipse cells')
    parser.add_argument('--I0', type=float, default=0.0,
                       help='Initial infected cells')
    parser.add_argument('--R0', type=float, default=0.0,
                       help='Initial refractory cells')
    parser.add_argument('--V0', type=float, default=1.0,
                       help='Initial virions')
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Simulation mode')
    
    # Single simulation mode
    single_parser = subparsers.add_parser('single', help='Run single simulation')
    single_parser.add_argument('--name', type=str, default='single_sim',
                              help='Name for simulation')
    
    # Parameter sweep mode
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep')
    sweep_parser.add_argument('--name_prefix', type=str, default='sweep',
                             help='Prefix for sweep simulation names')
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Parse parameter values (support comma-separated lists for sweeps)
    def parse_param_values(param_str: str) -> List[float]:
        """Parse parameter string to list of float values."""
        return [float(x.strip()) for x in param_str.split(',')]
    
    # Create deployment manager
    deployment = TEIRVDeployment(args.output_dir)
    
    # Create base configuration
    config = TEIRVSimulationConfig(
        t_max=args.t_max,
        dt=args.dt,
        method=args.method,
        switching_threshold=args.threshold
    )
    
    # Create initial conditions
    initial_conditions = TEIRVInitialConditions(
        T=args.T0,
        E=args.E0,
        I=args.I0,
        R=args.R0,
        V=args.V0
    )
    
    if args.mode == 'single':
        # Parse single parameter values
        parameters = TEIRVParameters(
            beta=parse_param_values(args.beta)[0],
            phi=parse_param_values(args.phi)[0],
            rho=parse_param_values(args.rho)[0],
            k=parse_param_values(args.k)[0],
            delta=parse_param_values(args.delta)[0],
            pi=parse_param_values(args.pi)[0],
            c=parse_param_values(args.c)[0]
        )
        
        # Run single simulation
        deployment.run_single_simulation(
            parameters=parameters,
            initial_conditions=initial_conditions,
            config=config,
            name=args.name
        )
    
    elif args.mode == 'sweep':
        # Parse parameter ranges
        parameter_ranges = {}
        param_args = {
            'beta': args.beta,
            'phi': args.phi,
            'rho': args.rho,
            'k': args.k,
            'delta': args.delta,
            'pi': args.pi,
            'c': args.c
        }
        
        for param_name, param_str in param_args.items():
            values = parse_param_values(param_str)
            if len(values) > 1:
                parameter_ranges[param_name] = values
        
        if not parameter_ranges:
            print("❌ No parameter ranges specified for sweep")
            return
        
        # Use first value of each parameter as base
        base_parameters = TEIRVParameters(
            beta=parse_param_values(args.beta)[0],
            phi=parse_param_values(args.phi)[0],
            rho=parse_param_values(args.rho)[0],
            k=parse_param_values(args.k)[0],
            delta=parse_param_values(args.delta)[0],
            pi=parse_param_values(args.pi)[0],
            c=parse_param_values(args.c)[0]
        )
        
        # Run parameter sweep
        deployment.run_parameter_sweep(
            base_parameters=base_parameters,
            base_initial_conditions=initial_conditions,
            base_config=config,
            parameter_ranges=parameter_ranges,
            name_prefix=args.name_prefix
        )
    
    # Save results and create plots
    deployment.save_results()
    deployment.create_summary_plots()
    
    print(f"\n✅ JSF TEIRV deployment completed")
    print(f"📁 Results saved to: {deployment.output_dir}")


if __name__ == '__main__':
    main()