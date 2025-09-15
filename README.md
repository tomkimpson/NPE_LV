# Neural Posterior Estimation for Viral Dynamics (TEIRV Model)

This project implements **Neural Posterior Estimation (NPE)** for parameter inference in viral dynamics using the TEIRV (Target-Eclipsed-Infectious-Refractory-Virion) model. The implementation replaces particle filtering methods with efficient neural posterior estimation for analyzing COVID-19 RT-PCR data.

## Overview

The TEIRV model describes viral infection dynamics as a continuous-time Markov chain with seven reactions:

1. **Infection**: T + V → E at rate β×T×V×10⁻⁹
2. **IFN protection**: T + I → R at rate φ×T×I×10⁻⁵  
3. **Reversion**: R → T at rate ρ×R
4. **Progression**: E → I at rate k×E
5. **Cell death**: I → ∅ at rate δ×I
6. **Viral production**: I → I + V at rate π×I
7. **Viral clearance**: V → ∅ at rate c×V

Given RT-PCR observations with detection limits and noise, we use NPE to infer the posterior distribution over parameters θ = (β, π, δ, φ, ρ, V₀).

## Project Structure

```
NPE_LV/
├── data/                              # Data directory
│   ├── raw/                           # Raw, unprocessed datasets
│   ├── processed/                     # Cleaned, transformed data ready for training
│   └── generated/                     # Data generated for simulation-based inference
├── models/                            # Model directory
│   ├── checkpoints/                   # Intermediate model weights during training
│   └── production/                    # Final, trained models ready for inference
├── notebooks/                         # Jupyter notebooks
│   ├── exploratory/                   # Initial data analysis, prototyping
│   └── results_analysis/              # Model performance analysis, visualizations
├── src/                               # Core source code (simplified flat structure)
│   ├── teirv_simulator.py             # Gillespie algorithm for TEIRV model
│   ├── teirv_data_generation.py       # NPE training data generation
│   ├── teirv_inference.py             # Neural posterior estimation training/inference
│   ├── clinical_data.py               # Clinical data loading and processing
│   ├── teirv_utils.py                 # TEIRV utilities, priors, and helpers
│   └── legacy/                        # Legacy implementations for reference
├── scripts/                           # Executable scripts
│   ├── slurm/                         # Slurm job submission scripts
│   │   ├── run_teirv_workflow.sh      # Main workflow script
│   │   ├── resume_teirv_training.sh   # Resume training script
│   │   └── workflows/                 # Additional workflow scripts
│   └── local/                         # Local utility scripts
│       ├── TEIRV_workflow.py          # TEIRV workflow implementation
│       ├── deploy_jsf_teirv.py        # JSF TEIRV deployment
│       └── teirv_jsf_examples.py      # Examples and demos
├── configs/                           # Configuration files
│   ├── model_configs/                 # Model hyperparameters
│   │   └── teirv_default.yaml         # Default TEIRV configuration
│   ├── training_configs/              # Training parameters
│   │   └── npe_default.yaml           # Default NPE training config
│   └── data_configs/                  # Data paths, processing parameters
│       ├── clinical_data.yaml         # Clinical data configuration
│       └── synthetic_data.yaml        # Synthetic data generation config
├── results/                           # Results and outputs
│   ├── experiments/                   # Logs, metrics, plots from experiments
│   │   ├── production_runs/           # Production experiment results
│   │   ├── training_logs/             # SBI training logs (tensorboard)
│   │   └── slurm_outputs/             # Slurm job outputs
│   ├── predictions/                   # Model predictions and outputs
│   └── reports/                       # Final reports, summary documents
├── tests/                             # Test files
├── external/                          # External repositories (preserved)
│   └── JSFGermano2024/                # Clinical data source
├── papers/                            # Research papers (preserved)
├── sandbox/                           # Experimental code (preserved)
├── development_notes/                 # Development documentation (renamed from docs/)
│   ├── phase3_clinical_integration.md
│   ├── teirv_implementation_plan.md
│   └── checkpoint_*.md
├── .gitignore                         # Git ignore file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
└── environment.yml                    # Conda environment specification
```

## Installation

### Option 1: Using Conda (Recommended)

1. **Clone with submodules**:
   ```bash
   git clone --recursive https://github.com/your-repo/NPE_LV.git
   cd NPE_LV
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate NPE_LV
   ```

3. **Install project as package**:
   ```bash
   pip install -e .
   ```

### Option 2: Using pip

1. **Clone with submodules**:
   ```bash
   git clone --recursive https://github.com/your-repo/NPE_LV.git
   cd NPE_LV
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Option 3: Development Installation

1. **Clone with submodules**:
   ```bash
   git clone --recursive https://github.com/your-repo/NPE_LV.git
   cd NPE_LV
   ```

2. **Install with development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Initialize Clinical Data

If submodules weren't initialized automatically:
```bash
git submodule update --init --recursive
```

## Implementation Phases

### Phase 1: TEIRV Simulator 
- Gillespie algorithm implementation for 7-reaction TEIRV model
- Proper parameter scaling and reaction rate computation
- RT-PCR observation model with detection limits
- Comprehensive testing and validation

### Phase 2: NPE Training Pipeline 
- Custom prior distributions (mixed Uniform/log-Uniform)
- Neural posterior estimation with SBI library
- Training data generation with batch processing
- Corner plots and posterior visualization
- Model saving and loading infrastructure

### Phase 3: Clinical Data Integration 
- Clinical data loading from JSFGermano2024 repository
- Patient filtering and quality control
- Individual patient inference pipeline
- Comprehensive visualization and reporting
- Multi-patient batch processing

## Quick Start

### Option 1: Quick Demo
```bash
conda activate NPE_LV
python src/main.py demo  # 2-3 minutes
```

### Option 2: Complete Manual Workflow
```bash
conda activate NPE_LV

# Step 1: Generate training data (5-10 minutes)
python src/main.py generate \
    --n_samples 5000 \
    --output data/processed/manual_training.pkl

# Step 2: Train NPE model (15-30 minutes)
python src/main.py train \
    --data data/processed/manual_training.pkl \
    --output models/production/manual_model.pkl

# Step 3: Run clinical inference (2-5 minutes)
python src/main.py inference \
    --model models/production/manual_model.pkl \
    --output results/predictions/manual_results
```

### Option 3: Complete Pipeline
```bash
# Run complete end-to-end pipeline
python src/main.py full --n_samples 10000

# Run via Slurm (for HPC environments)
sbatch scripts/slurm/run_teirv_workflow.sh
```

### Option 4: Individual Commands
```bash
# Generate training data only
python src/main.py generate --n_samples 50000

# Train model only
python src/main.py train --data data/processed/training.pkl

# Run inference only
python src/main.py inference --model models/production/npe.pkl
```

## Clinical Data

The project uses real RT-PCR data from COVID patients:

- **Source**: JSFGermano2024 repository (Germano et al., 2024)
- **Patients**: 6 COVID patients with longitudinal RT-PCR measurements
- **Timeframe**: 14 daily measurements per patient
- **Format**: log₁₀ viral load with detection limits
- **Quality**: 85-93% detection rates across patients

| Patient ID | Timepoints | Detection Rate | Peak Viral Load |
|------------|------------|----------------|-----------------|
| 432192     | 14         | 85.7%         | 7.72           |
| 443108     | 14         | 92.9%         | 7.83           |
| 444391     | 14         | 85.7%         | 7.51           |
| Others     | 14         | 85-93%        | 7.09-7.85      |

## Model Parameters

The TEIRV model infers six key parameters:

- **β (infection rate)**: Target cell infection rate [0, 20]
- **π (virion production)**: Virions produced per infectious cell [200, 600]
- **δ (cell clearance)**: Infected cell death rate [1, 11]
- **φ (interferon protection)**: IFN-mediated protection rate [0, 15]
- **ρ (reversion rate)**: Refractory cell reversion rate [0, 1]
- **V₀ (initial virions)**: Initial viral load [1, 148] (log-uniform)

## Manual Workflow Details

### Step-by-Step Breakdown

#### Step 1: Generate Training Data
```bash
python src/main.py generate --n_samples 5000 --output data/processed/manual_training.pkl
```
**What happens:**
- Simulates 5,000 TEIRV viral dynamics trajectories using Gillespie algorithm
- Applies RT-PCR observation model with detection limits and noise
- Samples parameters from realistic prior distributions
- **Time**: 5-10 minutes
- **Output**: `data/manual_training.pkl` (~50MB)
- **Success indicator**: >95% simulation success rate

#### Step 2: Train NPE Model
```bash
python src/main.py train --data data/processed/manual_training.pkl --output models/production/manual_model.pkl
```
**What happens:**
- Loads synthetic training data from Step 1
- Trains Neural Spline Flow with 256 hidden units and 8 coupling transforms
- Uses validation-based early stopping
- **Time**: 15-30 minutes
- **Output**: `models/manual_model.pkl` (~10MB)
- **Success indicator**: Validation loss decreases and stabilizes (typically < -2.0)

#### Step 3: Clinical Inference
```bash
python src/main.py inference --model models/production/manual_model.pkl --output results/predictions/manual_results
```
**What happens:**
- Loads trained model and clinical patient data
- Filters patients by data quality (detection rate, peak viral load)
- Runs NPE inference on each patient (10,000 posterior samples)
- Generates comprehensive visualizations and numerical summaries
- **Time**: 2-5 minutes
- **Output**: Complete analysis in `manual_results/` directory
- **Success indicator**: Parameter estimates within reasonable biological ranges

### Expected Output Structure

After completing the manual workflow:

```
NPE_LV/
├── data/processed/
│   └── manual_training.pkl                    # 5K training samples (Step 1)
├── models/production/
│   └── manual_model.pkl                       # Trained NPE model (Step 2)
├── results/
│   ├── experiments/training_logs/             # Training logs from Step 2
│   └── predictions/manual_results/            # Clinical inference results (Step 3)
│       └── clinical_inference_[timestamp]/
│           ├── patient_432192/
│           │   ├── patient_432192_summary.txt      # Numerical results
│           │   ├── patient_432192_marginals.png    # Posterior marginals
│           │   ├── patient_432192_pairwise.png     # Parameter correlations
│           │   ├── patient_432192_corner.png       # Corner plot
│           │   ├── patient_432192_predictive.png   # Model validation
│           │   └── patient_432192_raw_data.png     # Clinical data
│           ├── patient_443108/
│           │   └── ... (same structure)
│           ├── [other patients]/
│           └── all_patients_comparison.png         # Cross-patient analysis
```

### Validation Checklist

After each step, verify:

**Step 1 - Data Generation:**
- [ ] Success rate > 95%
- [ ] Parameter ranges: β∈[0,20], π∈[200,600], δ∈[1,11], φ∈[0,15], ρ∈[0,1], V₀∈[1,148]
- [ ] Observation dimension = 15 (14 daily timepoints + padding)

**Step 2 - NPE Training:**
- [ ] Training and validation losses decrease over epochs
- [ ] Final validation loss typically < -2.0
- [ ] No NaN values in loss curves
- [ ] Model file size ~10MB

**Step 3 - Clinical Inference:**
- [ ] Parameter estimates within prior bounds
- [ ] Posterior predictive checks show reasonable agreement
- [ ] All patients analyzed successfully (6 COVID patients available)
- [ ] Visualization files generated without errors

### Troubleshooting

**Step 1 Issues:**
- Low success rate: Reduce `t_max` or adjust batch size
- Memory errors: Reduce `n_samples` or `batch_size`

**Step 2 Issues:**
- Training divergence: Reduce learning rate (`--learning_rate 1e-4`)
- Slow convergence: Increase `max_epochs` or reduce `batch_size`

**Step 3 Issues:**
- No suitable patients: Lower filtering criteria (`--min-detections 3`)
- Visualization errors: Check matplotlib backend, install additional dependencies

## Advanced Usage

### Python API

```python
from src.clinical_data import ClinicalStudy
from src.teirv_inference import TEIRVInference

# Load clinical data
study = ClinicalStudy()
good_patients = study.filter_patients(min_detections=5, min_peak_viral_load=2.0)

# Load or train NPE model
inference = TEIRVInference.load('models/production/my_model.pkl')

# Run inference on patient
patient_id = good_patients[0]
x_obs = study.loader.preprocess_for_npe(patient_id)
posterior_samples = inference.sample_posterior(x_obs, num_samples=10000)
```

### Model Training

```python
from src.teirv_data_generation import TEIRVDataGenerator
from src.teirv_inference import TEIRVInference

# Generate training data
generator = TEIRVDataGenerator(seed=42)
theta, x = generator.generate_batch(n_samples=10000, batch_size=1000)

# Train NPE
inference = TEIRVInference(observation_type='rt_pcr')
training_info = inference.train(theta, x, max_num_epochs=100)

# Save model
inference.save('models/production/my_teirv_model.pkl')
```

## Validation and Testing

### Quick Validation
```bash
# Validate all components quickly (30 seconds)
python scripts/quick_teirv_test.py

# Validate clinical data compatibility (30 seconds)  
python scripts/run_clinical_inference.py --validate-only

# Quick demo with small model (2-3 minutes)
python scripts/demo_clinical_inference.py
```

### Comprehensive Testing
```bash
# Full pipeline validation (15-20 minutes)
python scripts/test_teirv_pipeline.py

# Manual workflow validation
python scripts/generate_teirv_data.py --n_samples 100 --output data/test.pkl
python scripts/train_teirv_npe.py --data data/test.pkl --output models/test.pkl --max_epochs 5
python scripts/run_clinical_inference.py --model-path models/test.pkl --output-dir test_results
```

### Debugging Performance Issues
```bash
# Debug simulation performance
python scripts/debug_teirv_performance.py

# Test individual components
python scripts/test_teirv_simulator.py
```

## Performance Considerations

### Training Data Generation
- **Batch size**: Use 500-1000 for memory efficiency
- **Total samples**: 5,000-50,000 depending on accuracy requirements  
- **Parallel processing**: Batch generation uses efficient vectorization
- **Success rate**: >95% for reasonable parameter ranges

### NPE Training
- **Network architecture**: 256 hidden units, 8 coupling transforms
- **Training time**: 30-60 minutes on CPU for 10K samples
- **GPU acceleration**: Significant speedup available with CUDA
- **Model size**: ~10MB saved models

### Clinical Inference
- **Inference time**: <1 second per patient for 10K samples
- **Memory usage**: ~100MB for typical analyses
- **Scalability**: Handles hundreds of patients efficiently




## References

- Germano et al. (2024). https://arxiv.org/abs/2405.13239
- Tejero-Cantero et al. (2020). "sbi: A toolkit for simulation-based inference"
- Papamakarios et al. (2019). "Sequential Neural Likelihood"
- Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions"




## License

MIT License