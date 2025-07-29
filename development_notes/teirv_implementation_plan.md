# TEIRV Model Implementation Plan

## Overview

Transitioning from **Lotka-Volterra predator-prey model** to **TEIRV viral dynamics model** with Neural Posterior Estimation (NPE). This represents a significant increase in complexity:

- **Compartments**: 5 (T, E, I, R, V) vs 2 (prey, predator)
- **Reactions**: 7 vs 4
- **Data**: Real clinical RT-PCR vs synthetic observations
- **Priors**: LogNormal vs BoxUniform
- **Observability**: Partial (V only) vs full state

## TEIRV Model Structure

### Compartments
- **T**: Target cells (susceptible to infection)
- **E**: Eclipsed cells (newly infected, not yet producing virus)
- **I**: Infectious cells (actively producing virions)
- **R**: Refractory cells (temporarily resistant due to interferon)
- **V**: Virions (free virus particles)

### Reactions
1. **Infection**: T + V → E (rate: β·T·V)
2. **Progression**: E → I (rate: k·E)
3. **Cell clearance**: I → ∅ (rate: δ·I)
4. **Viral production**: I → I + V (rate: π·I)
5. **Viral clearance**: V → ∅ (rate: c·V)
6. **Interferon protection**: T + I → R (rate: Φ·I·T)
7. **Reversion**: R → T (rate: ρ·R)

### Parameters

#### Inferred Parameters (6)
| Parameter | Description | Prior |
|-----------|-------------|-------|
| β | Infection rate | LogNormal(log(2.5×10⁻⁹), 1.0) |
| π | Virion production rate | LogNormal(log(10²), 1.0) |
| δ | Infected cell clearance | LogNormal(log(0.5), 1.0) |
| Φ | Interferon protection rate | LogNormal(log(10⁻⁹), 2.0) |
| ρ | Reversion rate R→T | LogNormal(log(0.1), 1.0) |
| V(0) | Initial virion count | LogNormal(log(10³), 1.0) |

#### Fixed Parameters (2)
| Parameter | Description | Value |
|-----------|-------------|-------|
| k | E→I transition rate | 4 |
| c | Virion clearance rate | 10 |

### Initial Conditions
- T(0) = 8×10⁷
- E(0) = 1
- I(0) = 0
- R(0) = 0
- V(0) = inferred parameter

### Observation Model
- **Data**: RT-PCR Cycle Number (CN) → log₁₀ viral load
- **Transformation**: log₁₀ V = 11.35 - 0.25 × CN
- **Noise**: y_t ~ N(log₁₀ V(t), σ² = 1)
- **Detection limit**: log₁₀ V ≥ -0.65
- **Timepoints**: 14 observations per patient

## Implementation Plan

### Phase 1: Core TEIRV Simulator

#### 1.1 Gillespie Algorithm Extension
```python
# src/teirv_simulator.py
def gillespie_teirv(theta, initial_conditions, t_max, t_grid):
    """
    Implement 7-reaction TEIRV Gillespie algorithm
    
    Key differences from LV:
    - 5D state vector [T, E, I, R, V]
    - Bimolecular reactions (T+V, T+I)
    - Different parameter scales
    """
```

**Implementation notes**:
- Handle bimolecular reactions correctly
- Account for very different timescales (viral vs ecological)
- Efficient handling of large T population (8×10⁷)

#### 1.2 Prior Distributions
```python
# src/teirv_utils.py
def create_teirv_prior():
    """
    LogNormal priors for 6 parameters
    Much wider dynamic range than LV model
    """
```

#### 1.3 Observation Model
```python
# src/teirv_observation.py
def apply_observation_model(V_trajectory, sigma=1.0):
    """
    Convert V(t) to synthetic RT-PCR observations:
    1. log₁₀ transformation
    2. Gaussian noise
    3. Detection limit truncation
    """
```

### Phase 2: NPE Pipeline Adaptation

#### 2.1 Data Generation
```python
# src/teirv_data_generation.py
class TEIRVDataGenerator:
    """
    Generate (θ, y) training pairs:
    - Sample 6 parameters from LogNormal priors
    - Run TEIRV Gillespie simulation
    - Apply observation model to V(t) only
    - Handle detection limits and missing data
    """
```

#### 2.2 Neural Network Architecture
- **Input**: 14 timepoints (vs 101 for LV)
- **Output**: 6 parameters (vs 4 for LV)
- **Considerations**: May need larger networks due to increased complexity

#### 2.3 Training Pipeline
```python
# scripts/train_teirv_npe.py
"""
Train NPE on TEIRV model:
- Generate large training dataset
- Handle LogNormal parameter transformations
- Validate on synthetic data with known parameters
"""
```

### Phase 3: Clinical Data Integration

#### 3.1 Real Data Loader
```python
# src/clinical_data.py
def load_patient_data():
    """
    Load 6 patients × 14 timepoints of RT-PCR data
    Handle missing observations and detection limits
    """
```

#### 3.2 Patient-Specific Inference
```python
# scripts/fit_clinical_data.py
"""
For each patient:
1. Load RT-PCR time series
2. Sample from posterior using trained NPE
3. Generate parameter estimates and credible intervals
4. Compare with literature values
"""
```

#### 3.3 Validation
- **Simulation-based calibration**: Test posterior recovery
- **Clinical validation**: Compare to particle filter results
- **Predictive validation**: Forecast viral load trajectories

### Phase 4: Performance Optimization (if needed)

#### 4.1 JAX Implementation
```python
# src/jax_teirv_simulator.py
@jax.jit
def gillespie_teirv_jax(theta, ic, t_max):
    """
    JAX-compiled Gillespie for 10-100x speedup
    """

# Batch processing
gillespie_batch = jax.vmap(gillespie_teirv_jax, in_axes=(0, None, None))
```

#### 4.2 SBIJax Integration
```python
# Use sbi-jax for GPU-accelerated NPE training
# Particularly beneficial for large training datasets
```

## File Structure

```
NPE_LV/
├── src/
│   ├── teirv_simulator.py       # 7-reaction Gillespie algorithm
│   ├── teirv_utils.py          # LogNormal priors, utilities
│   ├── teirv_observation.py    # RT-PCR observation model
│   ├── teirv_data_generation.py # Training data generation
│   ├── teirv_inference.py      # NPE training and inference
│   └── clinical_data.py        # Real patient data loader
├── scripts/
│   ├── generate_teirv_data.py  # Generate training data
│   ├── train_teirv_npe.py      # Train NPE model
│   ├── fit_clinical_data.py    # Fit to patient data
│   └── teirv_workflow.py       # End-to-end pipeline
├── data/
│   ├── clinical/               # Patient RT-PCR data
│   └── teirv_training/         # Generated training data
└── docs/
    └── teirv_implementation_plan.md # This file
```

## Key Differences from Lotka-Volterra

| Aspect | Lotka-Volterra | TEIRV |
|--------|----------------|-------|
| **Compartments** | 2 (prey, predator) | 5 (T, E, I, R, V) |
| **Reactions** | 4 | 7 |
| **Parameters** | 4 (α, β, δ, γ) | 6 inferred + 2 fixed |
| **Priors** | BoxUniform | LogNormal |
| **Initial conditions** | Fixed (50, 100) | Mostly fixed + V(0) inferred |
| **Observations** | Full state | Partial (V only) |
| **Data** | Synthetic | Real clinical RT-PCR |
| **Timescale** | Days-weeks | Days |
| **Parameter scales** | ~0.01-1.0 | ~10⁻⁹ to 10⁷ |

## Success Metrics

1. **Technical**: Successful NPE training with good posterior recovery on synthetic data
2. **Clinical**: Reasonable parameter estimates on real patient data
3. **Validation**: Comparable or better results than particle filter benchmark
4. **Performance**: Feasible runtime for clinical application

## Risk Mitigation

1. **Complexity**: Start with simplified version, gradually add features
2. **Speed**: Implement JAX version if NumPy too slow
3. **Convergence**: Use proven SBI architectures, extensive validation
4. **Clinical data**: Careful preprocessing and missing data handling

---

*Implementation started: [DATE]*  
*Branch: teirv-implementation*  
*Based on paper: Jump-Switch-Flow for Non-Markovian stochastic PDMP models*