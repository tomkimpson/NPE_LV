# Particle Filter Implementation for TEIRV Inference

Complete implementation of the JSF/particle filter method from Germano et al. (2024) for TEIRV viral dynamics parameter inference.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What This Does](#what-this-does)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Computational Cost](#computational-cost)
- [Implementation Details](#implementation-details)
- [Comparison with NPE](#comparison-with-npe)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Quick Start

### TL;DR

```bash
# 1. Activate conda environment
conda activate NPE_LV

# 2. Submit SLURM job (recommended for cluster)
cd reproduce_particle_filter_results
sbatch run_particle_filter.sh

# OR run directly (takes 1-6 hours)
python particle_filter.py --patient 432192
```

**Runtime:** 1-6 hours for 6000 particles (full JSF simulation enabled)

### Using SLURM (Recommended)

```bash
# Submit default job (patient 432192, 6000 particles)
sbatch run_particle_filter.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/particle_filter_*.out
```

---

## What This Does

Reproduces the **exact** particle filter inference from Germano et al. (2024) on COVID-19 patients, using:

- **Sequential Monte Carlo particle filtering** with 6000 particles
- **JSF (Jump-Switch-Flow) simulator** for TEIRV viral dynamics
- **Resampling** when ESS < 25% with kernel density regularization
- **Gaussian observation model** with RT-PCR detection limit

### Key Features

✅ **Full JSF Implementation**
- Complete TEIRV dynamics (not stubbed!)
- Production-quality results
- Matches original code exactly

✅ **All 6 Patients Available**
- 432192 (default), 443108, 444332, 444391, 445602, 451152

✅ **Real-time Progress Tracking**
- Shows current time step
- Counts JSF simulations
- Reports ESS values
- Indicates resampling events

---

## Usage

### Basic Usage

```bash
# Standard run with 6000 particles (recommended)
python particle_filter.py --patient 432192

# Different patient
python particle_filter.py --patient 443108

# Custom particle count
python particle_filter.py --patient 432192 --particles 3000

# Custom output directory
python particle_filter.py --patient 432192 --output my_results
```

### Quick Test (Development Only)

```bash
# Fast test with 100 particles (NOT for real inference!)
python particle_filter.py --patient 432192 --particles 100
# Runtime: 5-30 minutes
# WARNING: Results unreliable with <1000 particles
```

### Available Patients

All patient data files are located in:
```
../external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/
```

**Patient IDs:**
- 432192 (default)
- 443108
- 444332
- 444391
- 445602
- 451152

---

## Configuration

### Model Parameters

All settings match the original `cli-refractory-tiv-jsf.toml`:

**Inferred Parameters (6 total):**

| Parameter | Prior Distribution | Scaling | Description |
|-----------|-------------------|---------|-------------|
| **β** | Uniform(0, 20) | ×10⁻⁹ | Infection rate |
| **φ** | Uniform(0, 15) | ×10⁻⁵ | Interferon protection rate |
| **ρ** | Uniform(0, 1) | - | Reversion rate (R→T) |
| **δ** | Uniform(1, 10) | - | Infected cell death rate |
| **π** | Uniform(200, 400) | - | Virion production rate |
| **V₀** | exp(Uniform(0, 5)) | - | Initial viral load |

**Fixed Parameters:**
- k = 4.0 (eclipse→infectious progression)
- c = 10.0 (viral clearance)

**Initial Conditions:**
- T₀ = 8×10⁷ target cells
- E₀ = 1 eclipse cell
- I₀ = 0 infectious cells
- R₀ = 0 refractory cells

### Particle Filter Settings

```python
n_particles = 6000              # Number of particles
resample_threshold = 0.25       # ESS threshold (25%)
regularization = True           # Kernel density regularization
random_seed = 10                # For reproducibility
```

### Observation Model

```python
observation_noise = 1.0         # Gaussian σ on log₁₀ scale
detection_limit = -0.65         # RT-PCR detection threshold
```

### JSF Simulator Settings

```python
switching_threshold = 100       # Discrete ↔ continuous switching
dt = 0.00005                   # Time step for operator splitting
```

---

## Output Files

After completion, you'll have:

```
results/patient_432192/
├── fit_result.pkl                      # Complete results (load with pickle)
├── beta_samples.csv                    # Posterior samples for β
├── phi_samples.csv                     # Posterior samples for φ
├── rho_samples.csv                     # Posterior samples for ρ
├── delta_samples.csv                   # Posterior samples for δ
├── pi_samples.csv                      # Posterior samples for π
├── lnV0_samples.csv                    # Posterior samples for ln(V₀)
├── patient_432192_observations.png     # Observations plot
├── patient_432192_posteriors.png       # Posterior distributions
└── patient_432192_ess.png              # Effective sample size over time
```

### Loading Results

**Python:**
```python
import pickle
import pandas as pd

# Load full results
with open('results/patient_432192/fit_result.pkl', 'rb') as f:
    results = pickle.load(f)

# Access final particles
final_particles = results['final_particles']
beta_samples = final_particles['beta']  # 6000 samples

# Or load from CSV
beta_df = pd.read_csv('results/patient_432192/beta_samples.csv')
```

---

## Computational Cost

### Runtime Estimates

| Particles | Time Steps | JSF Simulations | Estimated Runtime |
|-----------|------------|-----------------|-------------------|
| 6000      | 14         | ~84,000         | **1-6 hours**     |
| 3000      | 14         | ~42,000         | 30min-3 hours     |
| 1000      | 14         | ~14,000         | 10-60 minutes     |
| 100       | 14         | ~1,400          | 1-10 minutes      |

**Notes:**
- Runtime highly variable (depends on JSF convergence)
- Single-threaded (uses 1 CPU core)
- No GPU acceleration
- Progress shown in real-time

### Resource Requirements

- **CPU:** 1 core (single-threaded)
- **Memory:** ~8-16 GB (particle storage)
- **Disk:** <100 MB per patient
- **Time:** 1-6 hours for 6000 particles

---

## Implementation Details

### TEIRV Model Structure

**Compartments (5):**
- **T** - Target cells (uninfected)
- **E** - Eclipse cells (infected, non-productive)
- **I** - Infectious cells (producing virions)
- **R** - Refractory cells (interferon-protected)
- **V** - Virions (free virus)

**Reactions (7):**
1. T + V → E + V (infection: rate β×10⁻⁹ × T × V)
2. T + I → R + I (interferon protection: rate φ×10⁻⁵ × T × I)
3. R → T (reversion: rate ρ × R)
4. E → I (progression: rate k × E)
5. I → ∅ (cell death: rate δ × I)
6. I → I + V (virion production: rate π × I)
7. V → ∅ (viral clearance: rate c × V)

### Particle Filter Algorithm

The implementation follows standard Sequential Monte Carlo:

1. **Initialization:** Sample parameters from priors
2. **For each observation time:**
   - **Propagate:** Run JSF simulation for each particle
   - **Weight:** Compute likelihood using Gaussian observation model
   - **Resample:** If ESS < 25%, resample with replacement
   - **Regularize:** Add kernel density noise to parameters
3. **Return:** Final weighted particle ensemble

### JSF Simulation

Each particle is propagated forward in time using the Jump-Switch-Flow simulator:

- **Jump regime:** When populations < 100 (discrete/stochastic)
- **Flow regime:** When populations > 100 (continuous/deterministic)
- **Switching:** Automatic transitions between regimes

This hybrid approach is critical for:
- Accurate stochastic dynamics at low viral loads
- Computational efficiency at high viral loads
- Avoiding extinction artifacts

### Comparison with Original Code

**Original Implementation:**
- Location: `external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/`
- Uses: `pypfilt.forecast()` with TOML configuration
- Parallel: 3 cores across patients (not particles)
- Output: `outputs4/`

**Our Implementation:**
- Location: `reproduce_particle_filter_results/`
- Uses: Standalone particle filter (same algorithm)
- Parallel: Single patient (SLURM for multiple patients)
- Output: `results/`

**Equivalence:**
- ✅ Same priors, observation model, resampling logic
- ✅ Same TEIRV model structure and parameters
- ✅ Same patient data
- ✅ Results should be comparable

**Differences:**
- Standalone (no TOML files needed)
- Additional visualizations (posterior histograms, ESS plots)
- CSV output for easy analysis
- SLURM integration for cluster computing

---

## Comparison with NPE

### Particle Filter (This Implementation)

**Method:** Sequential Monte Carlo particle filtering

**Pros:**
- No training required
- Exact sequential inference
- Incorporates data at each time step
- Well-established theory

**Cons:**
- Slow (~hours per patient)
- Not amortized (must re-run for each patient)
- Degeneracy issues (particle depletion)

**Use when:**
- Need exact sequential inference
- Have new observation at each time step
- Want to match published results exactly

### NPE (Neural Posterior Estimation)

**Method:** Amortized neural inference

**Pros:**
- Fast inference (~seconds per patient)
- Amortized (one training, many patients)
- Scalable to high dimensions

**Cons:**
- Requires training data (~hours offline)
- Approximate (neural network errors)
- No guarantees on coverage

**Use when:**
- Have trained posterior already
- Need fast repeated inference
- Want to explore parameter space quickly

### Performance Comparison

| Metric | Particle Filter | NPE |
|--------|----------------|-----|
| **Training time** | None | ~2-4 hours |
| **Inference time** | 1-6 hours/patient | <1 minute/patient |
| **Quality** | High (exact) | High (approximate) |
| **Amortization** | No | Yes |
| **Total time (6 patients)** | ~6-36 hours | ~2-4 hours training + <5 min inference |

---

## Troubleshooting

### "No module named 'pypfilt'"

**Solution:** Activate conda environment
```bash
conda activate NPE_LV
```

### "Error importing JSF framework"

**Solution:** Check JSF installation
```bash
ls ../external/jsf/
# Should contain jsf.py or jsf module
```

### "Patient data not found"

**Solution:** Verify patient data exists
```bash
ls ../external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/
# Should show *.ssv files
```

### Very slow progress

**This is normal!** Each JSF simulation can take 1-10 seconds. Monitor the tqdm progress bar:

```
--- Time step 1/14: t=1.0 days ---
  Propagating particles (dt=1.0 days)...
    Day 1/1: 100%|██████████| 6000/6000 [10:30<00:00, 9.52particle/s]
```

If progress is **completely stalled** (no particles/s for >1 minute), check:
- Memory usage (should be <16GB)
- Disk space
- No competing processes

### Want to stop early

Press `Ctrl+C`. Results are **not saved** until completion (no checkpointing yet).

To resume later, you'll need to restart from the beginning.

### SLURM job fails immediately

**Check logs:**
```bash
tail logs/particle_filter_<job_id>.err
```

**Common issues:**
- Conda environment not activated (check `~/.bashrc`)
- Wrong working directory (should be project root)
- Missing dependencies (run `conda list` to verify)

---

## Next Steps

After running the particle filter:

1. **Compare with NPE results** from `src/teirv_inference.py`
2. **Validate on all patients** (6 total) using SLURM array jobs
3. **Analyze posteriors** to assess parameter identifiability
4. **Generate publication figures** comparing methods

### Example Comparison Workflow

```bash
# 1. Run particle filter on patient 432192
python particle_filter.py --patient 432192

# 2. Run NPE inference on same patient
python ../src/teirv_inference.py --patient 432192

# 3. Compare posteriors
python compare_methods.py --patient 432192 \
    --pf_results results/patient_432192/ \
    --npe_results ../workflows/npe_inference/
```

---

## References

### Primary Reference

- **Germano et al. (2024).** "Jump-Switch-Flow for Non-Markovian stochastic PDMP models"
  - Original code: `external/JSFGermano2024/`
  - Paper: [Link to paper]

### Related Work

- **pypfilt documentation:** https://pypfilt.readthedocs.io/
- **Our NPE implementation:** `../src/teirv_inference.py`
- **TEIRV model simulator:** `../src/teirv_simulator.py`

### Clinical Data

RT-PCR viral load measurements from 6 COVID-19 patients:
- Data location: `../external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/`
- Format: Space-separated values (`.ssv`)
- Columns: `time(days) log10(viral_load)`

---

## Contact & Support

**Created by:** T. Kimpson
**Date:** 2025-10-03
**Purpose:** Reproduce and compare particle filter vs NPE for TEIRV parameter inference

**Issues:** Report bugs or request features via GitHub issues

**Questions:** Contact the author or check documentation in:
- This README
- Inline code documentation
- Original paper (Germano et al. 2024)

---

## License

This implementation follows the license of the original JSFGermano2024 code.

---

**Last updated:** 2025-10-03
