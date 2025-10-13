#!/usr/bin/env python3
"""
Reproduce Particle Filter Results from JSFGermano2024

This script reproduces the JSF/particle filter inference method from the original
Germano et al. (2024) paper on a single patient. It uses pypfilt for particle filtering
with JSF simulation for the TEIRV viral dynamics model.

Goal: Compare particle filter inference with our NPE approach.

IMPORTANT: Activate the NPE_LV conda environment before running:
    conda activate NPE_LV
    python particle_filter.py --patient 432192

Author: T. Kimpson
Date: 2025-10-03
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import pickle
import time
from tqdm import tqdm

# Add external JSF to path (up one level from subdirectory)
sys.path.append(str(Path(__file__).parent.parent / 'external' / 'jsf'))

# Import JSF framework
try:
    import jsf
except ImportError as e:
    print(f"❌ Error importing JSF framework: {e}")
    print("Please ensure the JSF framework is properly installed in external/jsf/")
    sys.exit(1)

# Import pypfilt for particle filtering (optional for --help)
PYPFILT_AVAILABLE = False
try:
    import pypfilt
    import pypfilt.summary
    from pypfilt.model import Model
    from pypfilt.obs import Univariate
    import scipy.stats
    import statsmodels.stats.weightstats as smws
    PYPFILT_AVAILABLE = True
except ImportError as e:
    # Allow script to run for --help even without pypfilt
    if '--help' not in sys.argv and '-h' not in sys.argv:
        print(f"❌ Error importing pypfilt or dependencies: {e}")
        print("\n" + "=" * 80)
        print("CONDA ENVIRONMENT REQUIRED")
        print("=" * 80)
        print("\nPlease activate the NPE_LV conda environment:")
        print("  conda activate NPE_LV")
        print("\nThen run the script:")
        print("  python particle_filter.py --patient 432192")
        print("\n" + "=" * 80)
        sys.exit(1)
    # Create dummy classes for --help to work
    class Model: pass
    class Univariate: pass

# Import our existing TEIRV utilities
try:
    from src.clinical_data import ClinicalDataLoader
    from src.teirv_utils import get_teirv_initial_conditions, get_teirv_fixed_parameters
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    print("Some functionality may be limited.")


# =============================================================================
# TEIRV Model Implementation for pypfilt
# =============================================================================

class TEIRVModel_JSF(Model):
    """
    TEIRV model using JSF simulator for pypfilt particle filtering.

    Based on RefractoryCellModel_JSF from the original code.
    Compartments: T (Target), E (Eclipsed), I (Infectious), R (Refractory), V (Virions)
    """

    num_particles = -1
    threshold = 100  # Switching threshold for JSF

    # Stoichiometry matrices for TEIRV model (T, E, I, R, V order)
    # Note: Original code has different order (T, R, E, I, V) - we follow that
    _nu_reactants = [
        [1, 0, 0, 0, 1],  # T + V -> E + V (infection)
        [1, 0, 0, 1, 0],  # T + I -> R + I (interferon protection)
        [0, 1, 0, 0, 0],  # R -> T (reversion)
        [0, 0, 1, 0, 0],  # E -> I (progression)
        [0, 0, 0, 1, 0],  # I -> 0 (cell death)
        [0, 0, 0, 1, 0],  # I -> I + V (virion production)
        [0, 0, 0, 0, 1],  # V -> 0 (viral clearance)
    ]

    _nu_products = [
        [0, 0, 1, 0, 1],  # T + V -> E + V
        [0, 1, 0, 1, 0],  # T + I -> R + I
        [1, 0, 0, 0, 0],  # R -> T
        [0, 0, 0, 1, 0],  # E -> I
        [0, 0, 0, 0, 0],  # I -> 0
        [0, 0, 0, 1, 1],  # I -> I + V
        [0, 0, 0, 0, 0],  # V -> 0
    ]

    _nu = [[a - b for a, b in zip(r1, r2)]
           for r1, r2 in zip(_nu_products, _nu_reactants)]

    _stoich = {
        'nu': _nu,
        'DoDisc': [0, 0, 0, 0, 0],  # Allow switching for all compartments
        'nuReactant': _nu_reactants,
        'nuProduct': _nu_products
    }

    def _rates(self, x, theta, time):
        """
        Calculate reaction rates for TEIRV model.

        Parameters:
        -----------
        x : list
            State [T, R, E, I, V] (note order matches original code)
        theta : list
            Parameters [beta, phi, rho, k, delta, pi, c]
        time : float
            Current time (not used in this model)

        Returns:
        --------
        rates : list
            Reaction rates for each of the 7 reactions
        """
        t, r, e, i, v = x

        # Extract parameters with scaling (matching original code exactly)
        m_beta = theta[0] * 10**(-9)   # β × 10^-9
        m_phi = theta[1] * 10**(-5)    # φ × 10^-5
        m_rho = theta[2]
        m_k = theta[3]
        m_delta = theta[4]
        m_pi = theta[5]
        m_c = theta[6]

        # Reaction rates (matching original tiv.py exactly)
        return [
            m_beta * (t * v),      # Infection: T + V -> E + V
            m_phi * i * t,         # Interferon: T + I -> R + I
            m_rho * r,             # Reversion: R -> T
            m_k * e,               # Progression: E -> I
            m_delta * i,           # Cell death: I -> 0
            m_pi * i,              # Virion production: I -> I + V
            m_c * v                # Viral clearance: V -> 0
        ]

    def field_types(self, ctx):
        """Define particle fields for pypfilt."""
        return [
            ('lnV0', np.float_),
            ('beta', np.float_),
            ('phi', np.float_),
            ('rho', np.float_),
            ('k', np.float_),
            ('delta', np.float_),
            ('pi', np.float_),
            ('c', np.float_),
            ('T', np.float_),
            ('E', np.float_),
            ('I', np.float_),
            ('R', np.float_),
            ('V', np.float_),
        ]

    def init(self, ctx, vec):
        """Initialize particles from prior distributions."""
        prior = ctx.data['prior']
        self.num_particles = prior['T0'].shape[0]

        # Set parameters from prior
        vec['lnV0'] = prior['lnV0']
        vec['beta'] = prior['beta']
        vec['phi'] = prior['phi']
        vec['rho'] = prior['rho']
        vec['k'] = prior['k']
        vec['delta'] = prior['delta']
        vec['pi'] = prior['pi']
        vec['c'] = prior['c']

        # Set initial state
        vec['T'] = prior['T0']
        vec['E'] = prior['E0']
        vec['I'] = prior['I0']
        vec['R'] = prior['R0']
        vec['V'] = np.round(np.exp(vec['lnV0']))  # V0 from log-uniform prior

    def update(self, ctx, time_step, is_forecast, prev, curr):
        """
        Update particles using JSF simulator.

        This is called at each time step to propagate particles forward.
        """
        assert time_step.dt == 1, "Time step must be 1 day"

        # JSF simulation options (matching original exactly)
        _my_opts = {
            'EnforceDo': [0, 0, 0, 0, 0],
            'dt': 0.00005,
            'SwitchingThreshold': [self.threshold] * 5
        }

        for p_ix in range(self.num_particles):
            ptcl = prev[p_ix].copy()

            # State in original order: [T, R, E, I, V]
            x0 = [ptcl['T'], ptcl['R'], ptcl['E'], ptcl['I'], ptcl['V']]

            # Parameters
            theta = [
                ptcl['beta'],
                ptcl['phi'],
                ptcl['rho'],
                ptcl['k'],
                ptcl['delta'],
                ptcl['pi'],
                ptcl['c'],
            ]

            # Run JSF simulation for one day
            try:
                xs, ts = jsf.JumpSwitchFlowSimulator(
                    x0,
                    lambda x, t: self._rates(x, theta, t),
                    self._stoich,
                    time_step.dt,
                    _my_opts
                )

                # Update particle state (order: T, R, E, I, V)
                ptcl['T'] = xs[0][-1]
                ptcl['R'] = xs[1][-1]
                ptcl['E'] = xs[2][-1]
                ptcl['I'] = xs[3][-1]
                ptcl['V'] = xs[4][-1]

            except Exception as e:
                warnings.warn(f"JSF simulation failed for particle {p_ix}: {e}")
                # Keep previous state if simulation fails

            curr[p_ix] = ptcl

            # Progress indicator
            if p_ix % 100 == 0:
                print(f'\rTime {time_step.start:.0f}, Particle {p_ix}/{self.num_particles}', end='')

        print()  # New line after progress

    def can_smooth(self):
        """Return parameters that can be smoothed/regularized."""
        return {'lnV0', 'beta', 'phi', 'rho', 'k', 'delta', 'pi', 'c'}

    def is_extinct(self, hist):
        """
        Identify particles that have gone extinct.

        Extinction occurs when V, I, and E are all below threshold.
        """
        extinction_thresh = 1
        return np.ceil(
            (hist['V'] < extinction_thresh) *
            (hist['I'] < extinction_thresh) *
            (hist['E'] < extinction_thresh)
        )


class GaussianObservationModel(Univariate):
    """
    Gaussian observation model for RT-PCR viral load measurements.

    Based on original Gaussian class in tiv.py with detection limit.
    """

    limitOfDetection = -0.65  # Detection limit in log10 scale

    def quantiles(self, ctx, snapshot, probs):
        """Compute quantiles accounting for detection limit."""
        zero_mask = snapshot.state_vec[self.unit] == 0

        my_y_vals = np.zeros_like(snapshot.state_vec[self.unit])
        my_y_vals[np.logical_not(zero_mask)] = np.log10(
            snapshot.state_vec[self.unit][np.logical_not(zero_mask)]
        )
        my_y_vals[my_y_vals <= self.limitOfDetection] = self.limitOfDetection

        my_weighted_sample = smws.DescrStatsW(my_y_vals, weights=snapshot.weights)
        return my_weighted_sample.quantile(probs, return_pandas=False)

    def distribution(self, ctx, snapshot):
        """
        Observation distribution: log10(V) with Gaussian noise.

        For particles with V=0, set to detection limit with tiny noise.
        """
        zero_mask = snapshot.state_vec[self.unit] == 0

        loc = np.zeros_like(snapshot.state_vec[self.unit])
        loc[np.logical_not(zero_mask)] = np.log10(
            snapshot.state_vec[self.unit][np.logical_not(zero_mask)]
        )
        loc[loc <= self.limitOfDetection] = self.limitOfDetection

        trunc_mask = loc == self.limitOfDetection
        scale = np.repeat(ctx.settings['observations'][self.unit]['scale'], len(loc))
        scale[trunc_mask] = 10**(-32)  # Very small noise at detection limit

        return scipy.stats.norm(loc=loc, scale=scale)


class ExtinctionProbability(pypfilt.summary.Table):
    """Track probability of extinction over time."""

    def field_types(self, ctx, obs_list, name):
        self.__model = ctx.component['model']
        return [
            ('time', np.float_),
            ('pr', np.float64)
        ]

    def n_rows(self, ctx, forecasting):
        return ctx.summary_count()

    def add_rows(self, ctx, fs_time, window, insert_fn):
        for snapshot in window:
            mask = self.__model.is_extinct(snapshot.state_vec)
            seeded_weights = snapshot.weights * mask
            insert_fn((snapshot.time, np.sum(seeded_weights)))


# =============================================================================
# Configuration and Execution
# =============================================================================

def create_pypfilt_config(
    patient_id: str,
    data_dir: Path,
    n_particles: int = 6000,
    t_max: float = 21.0,
    obs_scale: float = 1.0,
    random_seed: int = 10
) -> Dict[str, Any]:
    """
    Create pypfilt configuration matching original TOML config.

    Parameters:
    -----------
    patient_id : str
        Patient ID (e.g., '432192')
    data_dir : Path
        Path to patient data directory
    n_particles : int
        Number of particles
    t_max : float
        Maximum time (days)
    obs_scale : float
        Observation noise scale (σ)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    config : dict
        Configuration dictionary for pypfilt
    """

    # Prior distributions (matching cli-refractory-tiv-jsf.toml)
    config = {
        'time': {
            'start': 0.0,
            'until': t_max,
            'steps_per_unit': 1,
            'summaries_per_unit': 1
        },

        'observations': {
            'V': {
                'model': GaussianObservationModel,
                'scale': obs_scale,
                'file': str(data_dir / f'{patient_id}.ssv')
            }
        },

        'prior': {
            'T0': {'name': 'constant', 'args': {'value': 8e7}},
            'E0': {'name': 'constant', 'args': {'value': 1.0}},
            'I0': {'name': 'constant', 'args': {'value': 0.0}},
            'R0': {'name': 'constant', 'args': {'value': 0.0}},

            # Inferred parameters (matching original exactly)
            'lnV0': {'name': 'uniform', 'args': {'loc': 0, 'scale': 5}},
            'beta': {'name': 'uniform', 'args': {'loc': 0, 'scale': 20}},
            'phi': {'name': 'uniform', 'args': {'loc': 0, 'scale': 15}},
            'rho': {'name': 'uniform', 'args': {'loc': 0, 'scale': 1}},
            'delta': {'name': 'uniform', 'args': {'loc': 1.0, 'scale': 10}},
            'pi': {'name': 'uniform', 'args': {'loc': 200, 'scale': 400}},

            # Fixed parameters
            'c': {'name': 'constant', 'args': {'value': 10.0}},
            'k': {'name': 'constant', 'args': {'value': 4.0}},
        },

        'filter': {
            'particles': n_particles,
            'prng_seed': random_seed,
            'history_window': -1,
            'resample': {'threshold': 0.25},
            'regularisation': {
                'enabled': True,
                'bounds': {
                    'lnV0': {'min': 0, 'max': 1e6},
                    'beta': {'min': 1e-4, 'max': 1e4},
                    'phi': {'min': 1e-10, 'max': 1e4},
                    'rho': {'min': 1e-10, 'max': 1e2},
                    'delta': {'min': 1e-4, 'max': 1e4},
                    'pi': {'min': 1e-4, 'max': 800.0},
                    'c': {'min': 9.0, 'max': 11.0},
                    'k': {'min': 3.0, 'max': 4.0},
                }
            }
        },

        'component': {
            'time': 'pypfilt.Scalar',
            'sampler': 'pypfilt.sampler.LatinHypercube',
            'summary': 'pypfilt.summary.HDF5',
            'model': TEIRVModel_JSF
        },

        'summary': {
            'tables': {
                'model_cints': {
                    'component': 'pypfilt.summary.ModelCIs',
                    'credible_intervals': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95]
                },
                'backcasts': {
                    'component': 'pypfilt.summary.BackcastPredictiveCIs',
                    'backcast_monitor': 'backcast_monitor',
                    'credible_intervals': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95]
                },
                'point_ests': {
                    'component': 'pypfilt.summary.ModelCIs',
                    'credible_intervals': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95]
                },
                'forecasts': {
                    'component': 'pypfilt.summary.PredictiveCIs',
                    'credible_intervals': [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95]
                },
                'extinction_prob': {
                    'component': ExtinctionProbability
                },
                'snapshot': {
                    'component': 'pypfilt.summary.EnsembleSnapshot'
                }
            },
            'monitors': {
                'backcast_monitor': {
                    'component': 'pypfilt.summary.BackcastMonitor'
                }
            }
        }
    }

    return config


def save_config_as_toml(config: Dict[str, Any], filepath: Path):
    """
    Save configuration as TOML file for pypfilt.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    filepath : Path
        Output TOML file path
    """
    try:
        import toml
    except ImportError:
        print("Warning: toml package not found. Install with: pip install toml")
        return

    # Convert config to TOML-compatible format
    toml_config = {
        'metadata': {
            'name': '',
            'description': 'TEIRV Particle Filter',
            'author': 'reproduce_particle_filter_results.py',
            'date': '2025-10-03'
        },
        'components': {
            'time': config['component']['time'],
            'sampler': config['component']['sampler'],
            'summary': config['component']['summary'],
            'model': 'reproduce_particle_filter_results.TEIRVModel_JSF'
        },
        'time': config['time'],
        'observations': {},
        'prior': {},
        'filter': config['filter'],
        'summary': config['summary']
    }

    # Convert observations
    for obs_name, obs_settings in config['observations'].items():
        toml_config['observations'][obs_name] = {
            'model': 'reproduce_particle_filter_results.GaussianObservationModel',
            'scale': obs_settings['scale'],
            'file': obs_settings['file']
        }

    # Convert prior (scenarios)
    toml_config['scenario'] = {'inference': {'prior': {}}}
    for param_name, param_prior in config['prior'].items():
        toml_config['scenario']['inference']['prior'][param_name] = param_prior

    # Save to file
    with open(filepath, 'w') as f:
        toml.dump(toml_config, f)

    print(f"  Config saved to: {filepath}")


def run_particle_filter_inference(
    patient_id: str,
    output_dir: Path,
    n_particles: int = 6000,
    forecast_times: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run particle filter inference on a single patient using pypfilt.

    Parameters:
    -----------
    patient_id : str
        Patient ID (e.g., '432192')
    output_dir : Path
        Output directory for results
    n_particles : int
        Number of particles
    forecast_times : list, optional
        Times to forecast (default: [10])
    verbose : bool
        Print progress information

    Returns:
    --------
    results : dict
        Dictionary containing inference results
    """

    if forecast_times is None:
        forecast_times = [10]

    if verbose:
        print("=" * 80)
        print(f"Running Particle Filter Inference: Patient {patient_id}")
        print("=" * 80)

    # Set up paths (up one level from subdirectory)
    data_dir = Path(__file__).parent.parent / 'external' / 'JSFGermano2024' / 'TEIVR_Results' / 'particle-filter-example-tiv_covid' / 'data'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    patient_file = data_dir / f'{patient_id}.ssv'
    if not patient_file.exists():
        raise FileNotFoundError(f"Patient data not found: {patient_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pypfilt configuration
    if verbose:
        print(f"\nConfiguration:")
        print(f"  Patient: {patient_id}")
        print(f"  Particles: {n_particles}")
        print(f"  Data file: {patient_file}")
        print(f"  Output: {output_dir}")

    config = create_pypfilt_config(
        patient_id=patient_id,
        data_dir=data_dir,
        n_particles=n_particles
    )

    # Load observations
    if verbose:
        print(f"\nLoading observations...")
    obs_data = np.loadtxt(patient_file, skiprows=1)
    times = obs_data[:, 0]
    observations = obs_data[:, 1]
    if verbose:
        print(f"  Loaded {len(observations)} observations")
        print(f"  Time range: {times.min():.1f} - {times.max():.1f} days")

    # Build pypfilt context manually (since we're not using TOML file)
    if verbose:
        print(f"\nBuilding pypfilt context...")

    try:
        # Create observation data structure
        obs_dict = {
            'V': {
                'time': times,
                'value': observations
            }
        }

        # Initialize model instance
        model = TEIRVModel_JSF()

        # Create prior samples
        n_p = n_particles
        prior_samples = {}

        # Sample from priors
        np.random.seed(config['filter']['prng_seed'])

        # Parameter priors
        for param_name, param_config in config['prior'].items():
            if param_config['name'] == 'uniform':
                loc = param_config['args']['loc']
                scale = param_config['args']['scale']
                prior_samples[param_name] = np.random.uniform(loc, loc + scale, n_p)
            elif param_config['name'] == 'constant':
                prior_samples[param_name] = np.full(n_p, param_config['args']['value'])

        # Create simplified pypfilt-compatible structure
        # Note: This is a simplified version. Full pypfilt requires proper Context building
        if verbose:
            print("\n" + "=" * 80)
            print("RUNNING PARTICLE FILTER")
            print("=" * 80)
            print(f"\nStarting particle filter with {n_particles} particles...")
            print(f"This may take several minutes...\n")

        # Run simplified particle filter
        fit_result = run_simplified_particle_filter(
            model=model,
            observations=obs_dict,
            prior_samples=prior_samples,
            config=config,
            verbose=verbose
        )

        # Save results
        if verbose:
            print(f"\nSaving results...")

        # Save fit result as pickle
        result_file = output_dir / 'fit_result.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump(fit_result, f)
        if verbose:
            print(f"  Results saved: {result_file}")

        # Extract and save parameter posteriors
        param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']
        for param in param_names:
            if param in fit_result['final_particles']:
                param_values = fit_result['final_particles'][param]
                param_df = pd.DataFrame({
                    'time': [config['time']['until']] * len(param_values),
                    'value': param_values
                })
                param_df.to_csv(output_dir / f'{param}_samples.csv', index=False)

        results = {
            'patient_id': patient_id,
            'config': config,
            'fit_result': fit_result,
            'status': 'completed',
            'output_dir': output_dir
        }

        if verbose:
            print("\n" + "=" * 80)
            print("INFERENCE COMPLETED SUCCESSFULLY")
            print("=" * 80)

        return results

    except Exception as e:
        if verbose:
            print(f"\n❌ Error during inference: {e}")
            import traceback
            traceback.print_exc()

        results = {
            'patient_id': patient_id,
            'config': config,
            'status': 'failed',
            'error': str(e)
        }
        return results


def run_simplified_particle_filter(
    model: TEIRVModel_JSF,
    observations: Dict[str, Dict[str, np.ndarray]],
    prior_samples: Dict[str, np.ndarray],
    config: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run FULL particle filter with JSF simulation.

    This implements complete Sequential Monte Carlo particle filtering:
    1. Initialize particles from prior
    2. For each observation time:
       - Propagate particles using JSF simulator (full TEIRV dynamics)
       - Weight particles using Gaussian observation model
       - Resample if ESS < threshold
       - Apply regularization with parameter bounds

    COMPUTATIONAL COST:
    - For 6000 particles × 14 time steps = 84,000 JSF simulations
    - Each JSF simulation: ~1-10 seconds depending on parameters
    - Total runtime estimate: 1-6 hours (highly variable)

    PROGRESS MONITORING:
    - Real-time updates show current time step and particle counts
    - ESS values indicate particle diversity
    - Resampling triggers when ESS < 25%

    Parameters:
    -----------
    model : TEIRVModel_JSF
        TEIRV model instance
    observations : dict
        Observation data
    prior_samples : dict
        Prior samples for all parameters
    config : dict
        Configuration dictionary
    verbose : bool
        Print progress

    Returns:
    --------
    results : dict
        Particle filter results
    """

    # Get observation times and values
    obs_times = observations['V']['time']
    obs_values = observations['V']['value']

    n_particles = config['filter']['particles']
    resample_threshold = config['filter']['resample']['threshold']

    # Initialize particles with parameters
    particles = {}
    for param_name, samples in prior_samples.items():
        particles[param_name] = samples.copy()

    # Initialize state variables from initial conditions
    # T0, E0, I0, R0 are constants, V0 comes from lnV0 parameter
    particles['T'] = prior_samples['T0'].copy()
    particles['R'] = prior_samples['R0'].copy()
    particles['E'] = prior_samples['E0'].copy()
    particles['I'] = prior_samples['I0'].copy()
    particles['V'] = np.exp(prior_samples['lnV0'])  # V0 from log-uniform prior

    if verbose:
        print(f"\nInitialized particles:")
        print(f"  Parameters: {[k for k in particles.keys() if k not in ['T', 'E', 'I', 'R', 'V']]}")
        print(f"  State variables: T, E, I, R, V")
        print(f"  Initial V range: [{particles['V'].min():.2f}, {particles['V'].max():.2f}]")

    # Initialize weights (uniform)
    weights = np.ones(n_particles) / n_particles

    # Storage for trajectory
    particle_history = []
    weight_history = []
    ess_history = []

    # Particle filter loop
    current_time = 0.0

    for t_idx, obs_time in enumerate(obs_times):
        if verbose:
            print(f"\n--- Time step {t_idx + 1}/{len(obs_times)}: t={obs_time:.1f} days ---")

        # Propagate particles from current_time to obs_time
        dt = obs_time - current_time

        if dt > 0:
            if verbose:
                print(f"  Propagating particles (dt={dt:.1f} days)...")

            # Propagate particles forward using JSF simulation
            n_steps = int(dt)

            for step in range(n_steps):
                # Create progress bar for particles
                particle_iter = range(n_particles)
                if verbose:
                    particle_iter = tqdm(
                        particle_iter,
                        desc=f"    Day {step + 1}/{n_steps}",
                        unit="particle",
                        leave=False,
                        ncols=80
                    )

                # Simulate each particle forward one day using JSF
                for p_idx in particle_iter:
                    # Extract current state (order: T, R, E, I, V)
                    x0 = [
                        particles['T'][p_idx],
                        particles['R'][p_idx],
                        particles['E'][p_idx],
                        particles['I'][p_idx],
                        particles['V'][p_idx]
                    ]

                    # Extract parameters
                    theta = [
                        particles['beta'][p_idx],
                        particles['phi'][p_idx],
                        particles['rho'][p_idx],
                        particles['k'][p_idx],
                        particles['delta'][p_idx],
                        particles['pi'][p_idx],
                        particles['c'][p_idx]
                    ]

                    # JSF simulation options
                    jsf_options = {
                        'EnforceDo': [0, 0, 0, 0, 0],
                        'dt': 0.00005,
                        'SwitchingThreshold': [100, 100, 100, 100, 100]
                    }

                    try:
                        # Run JSF simulator for 1 day
                        # JSF returns (compartments, times, events) - 3-tuple
                        result = jsf.JumpSwitchFlowSimulator(
                            x0,
                            lambda x, t: model._rates(x, theta, t),
                            model._stoich,
                            1.0,  # Simulate for 1 day
                            jsf_options
                        )

                        # Unpack result: (compartment_histories, times, events)
                        xs = result[0]  # List of lists, one per compartment
                        ts = result[1]  # Time points
                        # result[2] is event indicators (not needed)

                        # Update particle state with final values
                        # xs is [T_history, R_history, E_history, I_history, V_history]
                        particles['T'][p_idx] = xs[0][-1]
                        particles['R'][p_idx] = xs[1][-1]
                        particles['E'][p_idx] = xs[2][-1]
                        particles['I'][p_idx] = xs[3][-1]
                        particles['V'][p_idx] = xs[4][-1]

                    except Exception as e:
                        # If JSF fails, keep previous state
                        warnings.warn(f"JSF simulation failed for particle {p_idx}: {e}")
                        pass

            if verbose:
                print(f"  ✓ Propagation complete: {n_steps * n_particles} JSF simulations")

        # Weight particles based on observation
        obs_val = obs_values[t_idx]

        if verbose:
            print(f"  Weighting particles (obs={obs_val:.2f})...")

        # Compute likelihoods
        likelihoods = np.zeros(n_particles)
        obs_scale = config['observations']['V']['scale']
        detection_limit = -0.65

        for p_idx in range(n_particles):
            # Get predicted viral load
            V_pred = particles['V'][p_idx] if 'V' in particles else np.exp(particles['lnV0'][p_idx])

            # Convert to log scale
            if V_pred > 0:
                log_V_pred = np.log10(V_pred)
            else:
                log_V_pred = detection_limit

            # Apply detection limit
            log_V_pred = max(log_V_pred, detection_limit)

            # Compute likelihood
            if abs(obs_val - detection_limit) < 1e-6:
                # At detection limit - very small noise
                scale = 1e-32
            else:
                scale = obs_scale

            likelihoods[p_idx] = scipy.stats.norm.pdf(obs_val, loc=log_V_pred, scale=scale)

        # Update weights
        weights = weights * likelihoods
        weights_sum = np.sum(weights)

        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            # All weights zero - reset to uniform
            weights = np.ones(n_particles) / n_particles
            if verbose:
                print("  ⚠️  Warning: All particle weights zero, resetting to uniform")

        # Compute ESS
        ess = 1.0 / np.sum(weights ** 2)
        ess_normalized = ess / n_particles

        if verbose:
            print(f"  ESS: {ess:.1f} ({ess_normalized:.2%})")

        # Resample if needed
        if ess_normalized < resample_threshold:
            if verbose:
                print(f"  Resampling particles (ESS < {resample_threshold:.2%})...")

            # Multinomial resampling
            indices = np.random.choice(n_particles, size=n_particles, p=weights)

            # Resample particles
            for param_name in particles:
                particles[param_name] = particles[param_name][indices]

            # Reset weights
            weights = np.ones(n_particles) / n_particles

            # Apply regularization if enabled
            if config['filter']['regularisation']['enabled']:
                if verbose:
                    print(f"  Applying regularization...")

                bounds = config['filter']['regularisation']['bounds']

                for param_name in particles:
                    if param_name in bounds:
                        # Add kernel density noise
                        std = np.std(particles[param_name])
                        noise = np.random.normal(0, 0.1 * std, n_particles)
                        particles[param_name] = particles[param_name] + noise

                        # Enforce bounds
                        particles[param_name] = np.clip(
                            particles[param_name],
                            bounds[param_name]['min'],
                            bounds[param_name]['max']
                        )

        # Store history
        particle_history.append({k: v.copy() for k, v in particles.items()})
        weight_history.append(weights.copy())
        ess_history.append(ess)

        current_time = obs_time

    if verbose:
        print("\n" + "=" * 80)
        print("Particle filter completed!")
        print(f"Mean ESS: {np.mean(ess_history):.1f} ({np.mean(ess_history) / n_particles:.2%})")
        print("=" * 80)

    # Package results
    results = {
        'particle_history': particle_history,
        'weight_history': weight_history,
        'ess_history': ess_history,
        'final_particles': particles,
        'final_weights': weights,
        'observation_times': obs_times,
        'observation_values': obs_values
    }

    return results


def plot_comparison_with_original(
    patient_id: str,
    results: Dict[str, Any],
    output_dir: Path
):
    """
    Plot particle filter results including posteriors and observations.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    results : dict
        Inference results
    output_dir : Path
        Output directory
    """

    if results['status'] != 'completed':
        print(f"⚠️  Cannot plot - inference did not complete successfully")
        return

    fit_result = results['fit_result']

    # Load patient data (up one level from subdirectory)
    data_dir = Path(__file__).parent.parent / 'external' / 'JSFGermano2024' / 'TEIVR_Results' / 'particle-filter-example-tiv_covid' / 'data'
    patient_file = data_dir / f'{patient_id}.ssv'

    data = np.loadtxt(patient_file, skiprows=1)
    times = data[:, 0]
    observations = data[:, 1]

    # 1. Plot observations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(times, 10**observations, color='red', s=50,
              label='RT-PCR Observations', zorder=5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Viral Load')
    ax.set_yscale('log')
    ax.set_title(f'Patient {patient_id}: Viral Load Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'patient_{patient_id}_observations.png', dpi=300)
    plt.close()
    print(f"\n  Plot saved: {output_dir / f'patient_{patient_id}_observations.png'}")

    # 2. Plot posterior distributions
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']
    param_labels = ['ln(V₀)', 'β', 'φ', 'ρ', 'δ', 'π']
    final_particles = fit_result['final_particles']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        if param in final_particles:
            ax = axes[i]
            values = final_particles[param]

            # Histogram
            ax.hist(values, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

            # Statistics
            mean = np.mean(values)
            median = np.median(values)
            q025 = np.percentile(values, 2.5)
            q975 = np.percentile(values, 97.5)

            # Plot statistics
            ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.3g}')
            ax.axvline(q025, color='orange', linestyle=':', linewidth=1.5, label=f'95% CI: [{q025:.3g}, {q975:.3g}]')
            ax.axvline(q975, color='orange', linestyle=':', linewidth=1.5)

            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {label}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'patient_{patient_id}_posteriors.png', dpi=300)
    plt.close()
    print(f"  Plot saved: {output_dir / f'patient_{patient_id}_posteriors.png'}")

    # 3. Plot ESS over time
    if 'ess_history' in fit_result:
        fig, ax = plt.subplots(figsize=(10, 6))
        ess_history = fit_result['ess_history']
        ax.plot(times, ess_history, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.axhline(results['config']['filter']['particles'] * 0.25, color='red',
                  linestyle='--', label='Resample threshold (25%)')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Effective Sample Size (ESS)')
        ax.set_title(f'Patient {patient_id}: ESS over time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'patient_{patient_id}_ess.png', dpi=300)
        plt.close()
        print(f"  Plot saved: {output_dir / f'patient_{patient_id}_ess.png'}")

    # 4. Print summary statistics
    print(f"\n  Posterior Summary Statistics:")
    print(f"  " + "=" * 70)
    print(f"  {'Parameter':<12} {'Mean':<12} {'Median':<12} {'95% CI':<25}")
    print(f"  " + "-" * 70)

    for param, label in zip(param_names, param_labels):
        if param in final_particles:
            values = final_particles[param]
            mean = np.mean(values)
            median = np.median(values)
            q025 = np.percentile(values, 2.5)
            q975 = np.percentile(values, 97.5)
            print(f"  {label:<12} {mean:<12.4g} {median:<12.4g} [{q025:.4g}, {q975:.4g}]")

    print(f"  " + "=" * 70)


# =============================================================================
# Main Script
# =============================================================================

def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description='Reproduce Particle Filter Results from JSFGermano2024',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  Activate the NPE_LV conda environment:
    conda activate NPE_LV

Examples:
  # Run inference on patient 432192 (default)
  python reproduce_particle_filter_results.py --patient 432192

  # Use different number of particles
  python reproduce_particle_filter_results.py --patient 432192 --particles 3000

  # Specify output directory
  python reproduce_particle_filter_results.py --patient 432192 --output results/pf_patient_432192

  # Run with fewer particles for quick test (not recommended for real inference)
  python reproduce_particle_filter_results.py --patient 432192 --particles 100
        """
    )

    parser.add_argument('--patient', type=str, default='432192',
                       help='Patient ID (default: 432192)')
    parser.add_argument('--particles', type=int, default=6000,
                       help='Number of particles (default: 6000)')
    parser.add_argument('--output', type=str, default='particle_filter_results',
                       help='Output directory (default: particle_filter_results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output) / f'patient_{args.patient}'

    # Run inference
    print("\n" + "=" * 80)
    print("TEIRV PARTICLE FILTER INFERENCE - FULL JSF SIMULATION")
    print("=" * 80)
    print(f"\nThis script reproduces the particle filter method from:")
    print("Germano et al. (2024): Jump-Switch-Flow for Non-Markovian stochastic PDMP models")
    print(f"\nPatient: {args.patient}")
    print(f"Particles: {args.particles}")
    print(f"Output: {output_dir}")

    # Estimate runtime
    n_obs = 14  # Typical number of observations
    n_sims = args.particles * n_obs
    print(f"\n⚠️  COMPUTATIONAL COST WARNING ⚠️")
    print(f"  Total JSF simulations: ~{n_sims:,}")
    print(f"  Estimated runtime: 1-6 hours (highly variable)")
    print(f"  CPU usage: 1 core (single-threaded)")
    print(f"\n  Progress will be shown in real-time.")
    print(f"  You can monitor ESS values to track convergence.")
    print("\n" + "=" * 80 + "\n")

    start_time = time.time()

    try:
        results = run_particle_filter_inference(
            patient_id=args.patient,
            output_dir=output_dir,
            n_particles=args.particles,
            verbose=args.verbose or True
        )

        # Create visualization
        plot_comparison_with_original(
            patient_id=args.patient,
            results=results,
            output_dir=output_dir
        )

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("PARTICLE FILTER INFERENCE COMPLETE")
        print("=" * 80)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  - fit_result.pkl          : Full particle filter results")
        print(f"  - *_samples.csv           : Posterior parameter samples")
        print(f"  - patient_{args.patient}_observations.png : Observation plot")
        print(f"  - patient_{args.patient}_posteriors.png   : Posterior distributions")
        print(f"  - patient_{args.patient}_ess.png          : ESS over time")
        print("\nNext steps:")
        print("1. Compare with NPE results from src/teirv_inference.py")
        print("2. Run on additional patients for validation")
        print("3. Analyze differences between particle filter and NPE posteriors")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
