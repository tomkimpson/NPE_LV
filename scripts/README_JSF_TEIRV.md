# JSF TEIRV Deployment Script

This directory contains scripts for deploying the TEIRV (Target-Eclipse-Infected-Refractory-Virus) epidemiological model using the modern JSF (Jump-Switch-Flow) framework.

## Overview

The deployment script bridges the old JSF TEIRV implementation with the new JSF framework, providing:
- Easy parameter configuration with validation
- Flexible initial conditions
- Choice of simulation methods (exact vs operator-splitting)
- Structured output management
- Batch processing capabilities
- Comprehensive plotting and analysis tools

## Files

- `deploy_jsf_teirv.py` - Main deployment script
- `teirv_jsf_examples.py` - Example usage scripts
- `README_JSF_TEIRV.md` - This documentation file

## Installation & Dependencies

### Prerequisites

1. **JSF Framework**: The new JSF framework must be installed in `external/jsf/`
2. **Python Dependencies**:
   ```bash
   pip install numpy pandas matplotlib scipy
   ```

### JSF Framework Setup

The script expects the JSF framework to be available at `external/jsf/`. If you have it elsewhere, modify the import path in the script.

## TEIRV Model Overview

The TEIRV model represents within-host viral dynamics with 5 compartments:

- **T**: Target cells (susceptible to infection)
- **E**: Eclipse cells (infected but not yet producing virus)
- **I**: Infected cells (producing virus)
- **R**: Refractory cells (protected by interferon)
- **V**: Virions (viral particles)

### Model Reactions

1. **Infection**: `T + V → E + V` (rate: β·T·V)
2. **Interferon Protection**: `T + I → R + I` (rate: φ·I·T)
3. **Reversion**: `R → T` (rate: ρ·R)
4. **Eclipse to Infected**: `E → I` (rate: k·E)
5. **Infected Cell Death**: `I → ∅` (rate: δ·I)
6. **Virus Production**: `I → I + V` (rate: π·I)
7. **Virus Clearance**: `V → ∅` (rate: c·V)

### Parameters

| Parameter | Description | Units | Default |
|-----------|-------------|--------|---------|
| β | Infection rate | 1/(virion·day) | 1×10⁻⁹ |
| φ | Interferon protection rate | 1/day | 1×10⁻⁵ |
| ρ | Reversion rate | 1/day | 0.1 |
| k | Eclipse to infected rate | 1/day | 4.0 |
| δ | Infected cell death rate | 1/day | 2.0 |
| π | Virus production rate | virions/(cell·day) | 400.0 |
| c | Virus clearance rate | 1/day | 10.0 |

## Usage

### Command Line Interface

#### Basic Single Simulation

```bash
# Run with default parameters
python deploy_jsf_teirv.py single

# Run with custom parameters
python deploy_jsf_teirv.py single --beta 2.0 --pi 500.0 --V0 10.0

# Use operator-splitting method
python deploy_jsf_teirv.py single --method operator-splitting --name "my_simulation"
```

#### Parameter Sweep

```bash
# Sweep over beta values
python deploy_jsf_teirv.py sweep --beta 0.5,1.0,2.0,5.0

# Multi-parameter sweep
python deploy_jsf_teirv.py sweep --beta 1.0,2.0 --pi 200.0,400.0,800.0

# Custom sweep with different method
python deploy_jsf_teirv.py sweep --beta 1.0,2.0 --delta 1.0,2.0,3.0 --method operator-splitting
```

#### Command Line Options

**Global Options:**
- `--output_dir`: Output directory for results (default: `teirv_jsf_results`)
- `--method`: Simulation method - `exact` or `operator-splitting` (default: `exact`)
- `--t_max`: Maximum simulation time in days (default: 10.0)
- `--dt`: Time step for operator splitting (default: 0.00005)
- `--threshold`: Switching threshold for discrete/continuous (default: 100.0)

**Parameter Options:**
- `--beta`: Infection rate (default: 1.0)
- `--phi`: Interferon protection (default: 1.0)
- `--rho`: Reversion rate (default: 0.1)
- `--k`: Eclipse to infected rate (default: 4.0)
- `--delta`: Infected cell death rate (default: 2.0)
- `--pi`: Virus production rate (default: 400.0)
- `--c`: Virus clearance rate (default: 10.0)

**Initial Condition Options:**
- `--T0`: Initial target cells (default: 8×10⁷)
- `--E0`: Initial eclipse cells (default: 0.0)
- `--I0`: Initial infected cells (default: 0.0)
- `--R0`: Initial refractory cells (default: 0.0)
- `--V0`: Initial virions (default: 1.0)

### Python API

#### Basic Usage

```python
from deploy_jsf_teirv import (
    TEIRVDeployment,
    TEIRVParameters,
    TEIRVInitialConditions,
    TEIRVSimulationConfig
)

# Create deployment manager
deployment = TEIRVDeployment("my_results")

# Set up parameters
parameters = TEIRVParameters(
    beta=2.0,
    phi=1.0,
    rho=0.1,
    k=4.0,
    delta=2.0,
    pi=400.0,
    c=10.0
)

# Set up initial conditions
initial_conditions = TEIRVInitialConditions(
    T=8e7,
    E=0.0,
    I=0.0,
    R=0.0,
    V=1.0
)

# Configure simulation
config = TEIRVSimulationConfig(
    t_max=10.0,
    method='exact',
    switching_threshold=100.0
)

# Run simulation
result = deployment.run_single_simulation(
    parameters=parameters,
    initial_conditions=initial_conditions,
    config=config,
    name="my_simulation"
)

# Save results and create plots
deployment.save_results()
deployment.create_summary_plots()
```

#### Parameter Sweep

```python
# Define parameter ranges
parameter_ranges = {
    'beta': [0.5, 1.0, 2.0, 5.0],
    'pi': [200.0, 400.0, 800.0]
}

# Run parameter sweep
sweep_results = deployment.run_parameter_sweep(
    base_parameters=parameters,
    base_initial_conditions=initial_conditions,
    base_config=config,
    parameter_ranges=parameter_ranges,
    name_prefix="parameter_sweep"
)
```

## Examples

### Running Examples

The `teirv_jsf_examples.py` script provides comprehensive usage examples:

```bash
# Run all examples
python teirv_jsf_examples.py all

# Run specific examples
python teirv_jsf_examples.py basic
python teirv_jsf_examples.py parameter
python teirv_jsf_examples.py method
python teirv_jsf_examples.py custom
python teirv_jsf_examples.py multi
```

### Example Scenarios

1. **Basic Simulation**: Simple run with default parameters
2. **Parameter Exploration**: Explore effect of different infection rates
3. **Method Comparison**: Compare exact vs operator-splitting methods
4. **Custom Scenario**: High initial viral load scenario
5. **Multi-Parameter Sweep**: Explore combinations of parameters

## Output Structure

The script generates organized output in the specified directory:

```
teirv_jsf_results/
├── teirv_results.json          # Raw simulation results
├── teirv_summary.png           # Summary plots for all compartments
└── [additional analysis files]
```

### Results Format

Each simulation result contains:
- **Parameters**: All model parameters used
- **Initial Conditions**: Starting compartment values
- **Configuration**: Simulation settings
- **Times**: Array of time points
- **Trajectories**: Array of compartment values over time
- **Metadata**: Simulation method, timing, etc.

## Simulation Methods

### Exact Method
- Mathematically exact implementation
- Handles both continuous and discrete regimes
- Slower but more accurate
- Recommended for: small systems, precision requirements

### Operator-Splitting Method
- Approximation using forward Euler steps
- Faster execution
- Good accuracy for most applications
- Recommended for: large parameter sweeps, long simulations

## Performance Considerations

- **Exact method**: ~0.1-1 seconds per simulation
- **Operator-splitting**: ~0.01-0.1 seconds per simulation
- **Memory usage**: ~1-10 MB per simulation
- **Parallel processing**: Not currently implemented (future enhancement)

## Troubleshooting

### Common Issues

1. **JSF Import Error**:
   ```
   Error importing JSF framework
   ```
   **Solution**: Ensure JSF framework is installed in `external/jsf/`

2. **Parameter Validation Error**:
   ```
   Parameter validation failed: beta must be positive
   ```
   **Solution**: Check parameter values are within valid ranges

3. **Simulation Failure**:
   ```
   Simulation failed: [error message]
   ```
   **Solution**: Try different time step (`--dt`) or switching threshold (`--threshold`)

### Performance Tips

1. **Use operator-splitting for large sweeps**: Much faster execution
2. **Reduce time resolution**: Use larger `dt` values when possible
3. **Limit simulation duration**: Use shorter `t_max` for exploratory runs
4. **Batch processing**: Use parameter sweeps instead of individual runs

## Integration with Existing Workflow

The deployment script is designed to integrate with the existing `TEIRV_workflow.py`:

```python
# Example integration
from deploy_jsf_teirv import TEIRVDeployment, TEIRVParameters

# Use in existing workflow
deployment = TEIRVDeployment("integration_results")
# ... run simulations ...
# Results compatible with existing analysis tools
```

## Future Enhancements

Planned improvements:
- [ ] Parallel processing for parameter sweeps
- [ ] Additional output formats (HDF5, CSV)
- [ ] Advanced plotting options
- [ ] Integration with uncertainty quantification
- [ ] Real-time monitoring for long simulations
- [ ] Automatic parameter optimization

## Support

For issues or questions:
1. Check this documentation
2. Review example scripts
3. Examine error messages for specific guidance
4. Validate parameter ranges and initial conditions

## Version History

- **v1.0** (2025-01-17): Initial implementation
  - Basic single simulation and parameter sweep functionality
  - Support for both exact and operator-splitting methods
  - Comprehensive validation and error handling
  - Example scripts and documentation