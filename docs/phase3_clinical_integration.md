# Phase 3: Clinical Data Integration

This document describes the clinical data integration phase of the TEIRV NPE implementation, which enables inference on real RT-PCR data from COVID patients.

## Overview

Phase 3 integrates the NPE pipeline with clinical RT-PCR data from the JSFGermano2024 repository, allowing us to:

1. Load and preprocess real patient viral load trajectories
2. Run NPE inference to estimate viral dynamics parameters for individual patients
3. Compare NPE results with the original particle filter approach
4. Analyze parameter variability across patient populations

## Clinical Data Sources

The clinical data comes from the external JSFGermano2024 repository:
- **Location**: `external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/`
- **Format**: Space-separated values (.ssv files)
- **Patients**: 6 COVID patients with RT-PCR measurements
- **Timeframe**: 14 daily measurements per patient
- **Units**: log₁₀ viral load

### Patient Dataset Summary

| Patient ID | Timepoints | Above Detection | Peak Viral Load | Detection Rate |
|------------|------------|-----------------|-----------------|----------------|
| 443108     | 14         | 13/14          | 7.83           | 92.9%          |
| 445602     | 14         | 12/14          | 7.85           | 85.7%          |
| 444391     | 14         | 12/14          | 7.51           | 85.7%          |
| 444332     | 14         | 13/14          | 7.09           | 92.9%          |
| 432192     | 14         | 12/14          | 7.72           | 85.7%          |
| 451152     | 14         | 12/14          | 7.32           | 85.7%          |

## Key Components

### 1. Clinical Data Infrastructure

**`src/clinical_data.py`** - Complete clinical data handling system:

- **`ClinicalDataLoader`**: Loads patient .ssv files with error handling
- **`ClinicalStudy`**: Manages multiple patients with filtering and summarization
- **Data validation**: Ensures compatibility with NPE pipeline
- **Preprocessing**: Applies detection limits and formatting for torch tensors

#### Key Features:
```python
# Load all patients
study = ClinicalStudy()

# Filter high-quality patients
good_patients = study.filter_patients(min_detections=5, min_peak_viral_load=2.0)

# Prepare for NPE inference
patient_data = study.prepare_for_inference(good_patients)
```

### 2. Clinical Inference Pipeline

**`scripts/run_clinical_inference.py`** - Complete clinical analysis workflow:

#### Features:
- **Flexible model loading**: Use pre-trained models or train on-demand
- **Patient filtering**: Quality-based selection criteria
- **Batch processing**: Analyze multiple patients efficiently
- **Comprehensive visualization**: Multiple plot types per patient
- **Results export**: Numerical summaries and publication-ready figures

#### Usage:
```bash
# Basic usage - analyze all good patients
python scripts/run_clinical_inference.py

# Analyze specific patient
python scripts/run_clinical_inference.py --patient-id 432192

# Use pre-trained model
python scripts/run_clinical_inference.py --model-path models/my_model.pkl

# Custom filtering criteria
python scripts/run_clinical_inference.py --min-detections 8 --min-peak-vl 3.0

# High-resolution sampling
python scripts/run_clinical_inference.py --num-samples 50000
```

### 3. Demo and Testing

**`scripts/demo_clinical_inference.py`** - Quick demonstration:
- Trains small NPE model for demo purposes
- Runs inference on single patient
- Creates basic visualization
- Perfect for testing and validation

## Clinical vs Synthetic Data

### Key Differences

| Aspect | Synthetic Data | Clinical Data |
|--------|----------------|---------------|
| Time grid | Flexible (typically 14 days) | Fixed 14 daily measurements |
| Noise model | Gaussian σ=1.0 | Real measurement errors |
| Detection limits | Simulated threshold | Actual RT-PCR sensitivity |
| Parameter ranges | Known truth | Unknown patient-specific |
| Validation | Ground truth available | No ground truth reference |

### Data Preprocessing Pipeline

1. **Load raw .ssv files** (time, log₁₀ viral load)
2. **Apply detection limit** (RT-PCR threshold: -0.65)
3. **Quality filtering** (minimum detections, peak viral load)
4. **Time grid alignment** (ensure compatibility with NPE)
5. **Tensor conversion** (PyTorch format for inference)

## Clinical Inference Results

### Parameter Interpretation

The TEIRV model parameters estimated from clinical data represent:

- **β (infection rate)**: Rate of target cell infection [0, 20]
- **π (virion production)**: Virions produced per infectious cell [200, 600]  
- **δ (cell clearance)**: Infected cell death rate [1, 11]
- **φ (interferon protection)**: IFN-mediated protection rate [0, 15]
- **ρ (reversion rate)**: Refractory cell reversion rate [0, 1]
- **V₀ (initial virions)**: Initial viral load [exp(0), exp(5)] ≈ [1, 148]

### Expected Clinical Ranges

Based on viral dynamics literature and the original particle filter results:

- **Early infection**: High β, low φ (rapid viral growth)
- **Peak phase**: High π, moderate δ (maximum viral production)
- **Resolution**: High φ, high δ (immune control)
- **Patient variability**: 2-3 orders of magnitude in some parameters

## Validation and Quality Control

### Data Quality Metrics

1. **Detection rate**: Fraction of measurements above RT-PCR limit
2. **Peak timing**: Day of maximum viral load
3. **Trajectory shape**: Monotonic decline vs. multi-peak patterns
4. **Duration**: Time from peak to clearance

### NPE Validation Checks

1. **Parameter bounds**: Estimates within prior support
2. **Posterior coverage**: Credible intervals contain reasonable values  
3. **Predictive accuracy**: Posterior predictions match observations
4. **Cross-patient consistency**: Similar dynamics across patients

### Comparison with Particle Filter

The original JSFGermano2024 paper used particle filtering. Key comparisons:

| Method | Computational Cost | Flexibility | Accuracy | Scalability |
|--------|-------------------|-------------|----------|-------------|
| Particle Filter | High (SMC sampling) | Limited | High | Poor |
| NPE | Low (neural network) | High | Comparable | Excellent |

## Usage Examples

### 1. Quick Clinical Demo

```bash
# Fast demo on single patient
conda activate NPE_LV
python scripts/demo_clinical_inference.py
```

### 2. Full Clinical Analysis

```bash
# Comprehensive analysis of all patients
python scripts/run_clinical_inference.py \
    --num-samples 20000 \
    --output-dir results/clinical_analysis_v1
```

### 3. High-Quality Patient Analysis

```bash
# Focus on patients with excellent data quality
python scripts/run_clinical_inference.py \
    --min-detections 10 \
    --min-peak-vl 4.0 \
    --num-samples 50000
```

### 4. Model Reuse

```bash
# Train once, reuse for multiple analyses
python scripts/run_clinical_inference.py \
    --model-path models/teirv_npe_clinical_20250617.pkl \
    --patient-id 432192
```

## Output Structure

Clinical inference generates comprehensive results:

```
clinical_results/
├── clinical_inference_20250617_143022/
│   ├── patient_432192/
│   │   ├── patient_432192_summary.txt      # Numerical results
│   │   ├── patient_432192_marginals.png    # Posterior marginals
│   │   ├── patient_432192_pairwise.png     # Parameter correlations
│   │   ├── patient_432192_corner.png       # Corner plot
│   │   ├── patient_432192_predictive.png   # Predictive check
│   │   └── patient_432192_raw_data.png     # Clinical data
│   ├── patient_443108/
│   │   └── ...
│   └── all_patients_comparison.png          # Cross-patient comparison
```

## Clinical Insights

### Population-Level Analysis

NPE enables efficient population-level studies:

1. **Parameter distributions** across patient cohorts
2. **Correlation analysis** between clinical outcomes and model parameters
3. **Subgroup identification** based on viral dynamics patterns
4. **Treatment response prediction** using parameter estimates

### Personalized Medicine Applications

Patient-specific parameter estimates enable:

1. **Treatment optimization**: Tailor interventions to viral dynamics
2. **Prognosis prediction**: Use early parameters to predict outcomes
3. **Drug dosing**: Optimize antiviral timing and dosage
4. **Risk stratification**: Identify high-risk patients early

## Next Steps: Phase 4

Phase 3 provides the foundation for Phase 4 optimizations:

1. **JAX implementation**: Speed up inference for large patient cohorts
2. **Real-time inference**: Deploy for clinical decision support
3. **Multi-site validation**: Test on additional patient datasets
4. **Treatment integration**: Incorporate antiviral interventions

## Troubleshooting

### Common Issues

1. **Missing clinical data**: Ensure JSFGermano2024 submodule is properly initialized
2. **Time grid mismatch**: Clinical data has 14 points, check preprocessing
3. **Memory issues**: Reduce batch sizes or number of posterior samples
4. **Slow inference**: Use pre-trained models instead of training from scratch

### Performance Tips

1. **Pre-train models**: Save and reuse NPE models across analyses
2. **Batch patients**: Process multiple patients in single script run
3. **Parallel inference**: Use multiple processes for large cohorts
4. **GPU acceleration**: Train on GPU, infer on CPU for clinical deployment

## Validation Status

- ✅ Clinical data loading and preprocessing
- ✅ NPE training on clinical-compatible synthetic data
- ✅ Individual patient inference pipeline
- ✅ Comprehensive visualization and reporting
- ✅ Multi-patient batch processing
- ✅ Quality control and validation checks
- ✅ Results export and documentation

Phase 3 is complete and ready for clinical applications!