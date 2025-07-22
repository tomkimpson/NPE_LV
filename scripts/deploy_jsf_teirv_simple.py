#!/usr/bin/env python3
"""
Simplified JSF TEIRV Simulation Script

This script runs a TEIRV (Target-Eclipse-Infected-Refractory-Virus) 
epidemiological model simulation using the JSF framework.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the external JSF package to the Python path
# Assumes the script is in a directory like 'project/scripts/' and JSF is in 'project/external/jsf/'
try:
    sys.path.append(str(Path(__file__).parent.parent / 'external' / 'jsf'))
    import jsf
    from jsf.types import SystemState
except ImportError as e:
    print(f"Error importing JSF framework: {e}")
    print("Please ensure the JSF framework is located in the correct directory.")
    sys.exit(1)

# ==============================================================================
# ⚙️ 1. CONFIGURE YOUR SIMULATION HERE
# ==============================================================================

# --- Model Parameters ---
MODEL_PARAMS = {
    'beta': 17e-9,       # Infection rate (1/virion/day)
    'phi': 10e-5,        # Interferon protection rate (1/day)
    'rho': 0.6,         # Reversion rate from R to T (1/day)
    'k': 4.0,           # Eclipse to infected transition rate (1/day)
    'delta': 5.0,       # Infected cell death rate (1/day)
    'pi': 400.0,        # Virus production rate (virions/infected_cell/day)
    'c': 10.0           # Virus clearance rate (1/day)
}

# --- Initial Conditions ---
INITIAL_CONDITIONS = {
    'T': 8e7,           # Target cells
    'E': 0.0,           # Eclipse cells
    'I': 0.0,           # Infected cells
    'R': 0.0,           # Refractory cells
    'V': 100.0          # Virions
}

# --- Simulation Settings ---
SIM_CONFIG = {
    't_max': 10.0,                    # Maximum simulation time (days)
    'method': 'exact',                # 'exact' or 'operator-splitting'
    'switching_threshold': 100.0,     # Threshold for discrete/continuous switching
    'dt': 0.00005                     # Time step for operator-splitting method
}

# ==============================================================================
# 🔬 2. MODEL DEFINITION AND SIMULATION FUNCTIONS
# ==============================================================================

def setup_stoichiometry() -> dict:
    """
    Defines the stoichiometry for the TEIRV model reactions.
    
    Returns:
        A dictionary containing the stoichiometry matrices for JSF.
    """
    # Reactions: T, E, I, R, V
    # 1. T + V -> E + V     (Infection)
    # 2. T + I -> R + I     (Interferon protection)
    # 3. R -> T             (Reversion)
    # 4. E -> I             (Eclipse to infected)
    # 5. I -> 0             (Infected cell death)
    # 6. I -> I + V         (Virus production)
    # 7. V -> 0             (Virus clearance)
    
    nu_reactants = [
        [1, 0, 0, 0, 1],  # T + V
        [1, 0, 1, 0, 0],  # T + I
        [0, 0, 0, 1, 0],  # R
        [0, 1, 0, 0, 0],  # E
        [0, 0, 1, 0, 0],  # I
        [0, 0, 1, 0, 0],  # I
        [0, 0, 0, 0, 1],  # V
    ]
    
    nu_products = [
        [0, 1, 0, 0, 1],  # E + V
        [0, 0, 1, 1, 0],  # I + R 
        [1, 0, 0, 0, 0],  # T
        [0, 0, 1, 0, 0],  # I
        [0, 0, 0, 0, 0],  # 0
        [0, 0, 1, 0, 1],  # I + V
        [0, 0, 0, 0, 0],  # 0
    ]
    
    # Net change matrix (nu) is calculated as products - reactants
    nu = [[p - r for p, r in zip(prod, react)] for prod, react in zip(nu_products, nu_reactants)]
    
    return {
        'nu': nu,
        'DoDisc': [0, 0, 0, 0, 0],
        'nuReactant': nu_reactants,
        'nuProduct': nu_products
    }


def calculate_rates(x: list, t: float, params: dict) -> list:
    """
    Calculates the reaction rates at a given state and time.
    
    Args:
        x: Current state vector [T, E, I, R, V].
        t: Current time (not used in this model, but required by JSF).
        params: Dictionary of model parameters.
        
    Returns:
        A list of reaction rates.
    """
    T, E, I, R, V = x
    
    p = params
    rates = [
        p['beta'] * T * V,       # Infection
        p['phi'] * I * T,        # Interferon protection
        p['rho'] * R,            # Reversion
        p['k'] * E,              # Eclipse to infected
        p['delta'] * I,          # Infected cell death
        p['pi'] * I,             # Virus production
        p['c'] * V               # Virus clearance
    ]
    return rates


def run_teirv_simulation(params: dict, initial_cond: dict, config: dict) -> tuple:
    """
    Runs the TEIRV simulation using the JSF framework.
    
    Args:
        params: Dictionary of model parameters.
        initial_cond: Dictionary of initial conditions.
        config: Dictionary of simulation settings.
        
    Returns:
        A tuple containing (times, trajectories) as NumPy arrays.
    """
    print("🚀 Starting TEIRV simulation...")
    
    # Get stoichiometry
    stoich = setup_stoichiometry()
    
    # Set initial state from the dictionary
    x0_list = [initial_cond[key] for key in ['T', 'E', 'I', 'R', 'V']]
    x0 = SystemState(x0_list)
    
    # Create a rate function that JSF can call (with only x and t as arguments)
    rate_func = lambda x, t: calculate_rates(x, t, params)
    
    # Prepare JSF options
    num_compartments = len(x0_list)
    jsf_options = {
        'dt': config['dt'],
        'SwitchingThreshold': [config['switching_threshold']] * num_compartments,
        'EnforceDo': [0] * num_compartments
    }
    
    # Run simulation based on the chosen method
    if config['method'] == 'exact':
        trajectory = jsf.jsf(
            x0=x0,
            rates=rate_func,
            stoich=stoich,
            t_max=config['t_max'],
            method='exact',
            config=jsf_options
        )
    else: # operator-splitting
        trajectory = jsf.JumpSwitchFlowSimulator(
            x0=x0,
            rates=rate_func,
            stoich=stoich,
            t_max=config['t_max'],
            options=jsf_options
        )
        
    # Extract and format results
    times = np.array(trajectory[1])
    trajectories = np.column_stack(trajectory[0])
    
    print(f"✅ Simulation complete. Generated {len(times)} time points.")
    
    return times, trajectories


def plot_results(times: np.ndarray, trajectories: np.ndarray, initial_cond: dict):
    """
    Generates and displays a plot of the simulation results.
    
    Args:
        times: Array of time points.
        trajectories: Array of species populations over time.
    """
    print("📊 Plotting results...")
    
    compartment_names = list(initial_cond.keys())
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    
    for i, name in enumerate(compartment_names):
        ax = axes[i]
        ax.plot(times, trajectories[:, i], color=colors[i], label=name)
        ax.set_title(f'Compartment: {name}')
        ax.set_ylabel('Population')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log') # Use log scale for better visualization
        ax.set_ylim(bottom=1) # Set bottom to 1 for log scale
        ax.legend()
        
    # Shared X-axis label
    for ax in axes:
        ax.set_xlabel('Time (days)')
        
    # Remove the empty subplot
    fig.delaxes(axes[5])
    
    fig.suptitle('TEIRV Model Simulation Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('teirv_jsf_results/predictions.png')
    plt.show()


# ==============================================================================
# 🏁 3. RUN THE SIMULATION AND VISUALIZE
# ==============================================================================

if __name__ == '__main__':
    # Run the simulation with the configurations defined at the top
    sim_times, sim_trajectories = run_teirv_simulation(
        params=MODEL_PARAMS,
        initial_cond=INITIAL_CONDITIONS,
        config=SIM_CONFIG
    )
    
    # Plot the results
    plot_results(sim_times, sim_trajectories, INITIAL_CONDITIONS)