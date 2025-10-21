# TEIRV NPE Training Data

This directory contains training datasets for the TEIRV Neural Posterior Estimation model.

## Training Dataset

### `training_200k_20251013.pkl` (13 MB)
- **Samples:** 200,000
- **Generated:** October 13, 2025
- **Source:** Job 5935080 (production run)
- **Parameters:** 6 (β, π, δ, φ, ρ, V₀)
- **Observations:** 10 dimensions (RT-PCR viral load, 1-day intervals over 10 days)
- **Time grid:** 10 days, dt=1.0 day (11 timepoints)
- **Observation type:** RT-PCR only (log₁₀ transformed viral loads)
- **Generation time:** ~24.3 hours
- **Status:** ✅ Production quality - currently used for all training

## Usage

The training dataset is configured in `configs/teirv_config.yaml`:

```yaml
data:
  source: "data/training_200k_20251013.pkl"  # Current production dataset
```

To generate new training data instead:

```yaml
data:
  source: "generate"
  output_name: "training_new.pkl"
  generation:
    n_samples: 200000  # Number of samples to generate
```

**Note:** Generating 200k samples takes ~24 hours. Adjust SLURM time allocation accordingly.

## Clinical Data

Patient clinical data is stored in the `clinical/` subdirectory. See `clinical/README.md` for details.
