# TEIRV NPE Posterior Quality Improvement Plan

**Date Created:** July 4, 2025  
**Branch:** teirv-implementation  
**Status:** Phase 2 - Training Optimization

## Problem Statement

The TEIRV Neural Posterior Estimation (NPE) workflow runs successfully end-to-end but produces **poor posterior quality**:

- **Broad, flat posteriors**: Parameter distributions are not peaked or narrow
- **Poor predictive plots**: Model predictions don't agree well with observed data
- **Lack of concentration**: Posteriors appear uninformative rather than concentrated around true values

## Current System Status ✅

### Working Components (Verified July 4, 2025)
- ✅ **End-to-end pipeline**: Complete workflow from data generation to clinical inference
- ✅ **Device compatibility**: GPU training and CPU/CUDA inference working
- ✅ **Implementation accuracy**: Code verified against original paper reference
- ✅ **Recent successful run**: Job 1992097 completed with 6 patients processed

### Implementation Verification Against Original Paper
Compared our implementation with `external/JSFGermano2024/TEIVR_Results/` and confirmed **100% consistency**:

| Component | Our Implementation | Paper Reference | Status |
|-----------|-------------------|-----------------|---------|
| **Observation Model** | `log₁₀V = 11.35 - 0.25×CN` | Same formula in `teirv_utils.py:315` | ✅ |
| **Noise Model** | `σ = 1.0`, detection limit `-0.65` | `scale = 1.0`, `limitOfDetection = -0.65` | ✅ |
| **Parameter Scaling** | `β×10⁻⁹`, `φ×10⁻⁵` | `m_beta*10**(-9)`, `m_phi*10**(-5)` | ✅ |
| **Priors** | Uniform(0,20), Uniform(200,600), etc. | Same bounds in `cli-refractory-tiv-jsf.toml` | ✅ |
| **Fixed Parameters** | `k=4`, `c=10` | Same in config file | ✅ |
| **Initial Conditions** | `T(0)=8e7`, `E(0)=1`, `I(0)=0`, `R(0)=0` | Same in config file | ✅ |

## Root Cause Analysis

### Primary Issue: Training Data Timespan Mismatch
**Problem**: Currently training on 14/15-day complete trajectories
**Should be**: Training on 10-day partial trajectories, predicting days 10-20

**Impact**: 
- Model learns to fit full trajectory rather than early infection dynamics
- Overfitting to later stages when viral dynamics are different
- Poor generalization to prediction phase

### Secondary Issues
1. **Training duration**: Only 38 epochs with early stopping (may need longer training)
2. **Validation metrics**: Limited posterior quality diagnostics during training
3. **Data quantity**: 50k samples may be insufficient for complex 6-parameter problem

## Implementation Plan

### Phase 1: Core Training Data Fix (High Priority) 🎯

#### Tasks:
1. **Modify data generation timespan**
   - Change `t_max=14.0` → `t_max=10.0` in data generation
   - Update observation grid to 11 points (0-10 days)
   - Files: `src/TEIRV/teirv_data_generation.py`, `scripts/TEIRV_workflow.py`

2. **Update predictive plotting**
   - Modify `_create_predictive_plot()` to predict days 10-20
   - Update credible interval ranges for new timespan
   - File: `scripts/TEIRV_workflow.py:379-541`

3. **Test shortened training data**
   - Generate new 10-day training dataset
   - Retrain NPE model with 10-day data
   - Evaluate posterior concentration improvement

4. **Add training diagnostics**
   - Monitor loss curves for convergence
   - Add posterior sample visualization during training
   - Track validation metrics over epochs

#### Expected Outcomes:
- **Well-concentrated posteriors** focused on early infection dynamics
- **Improved parameter identifiability** from critical growth phase
- **Better predictive accuracy** for days 10-20

#### Success Metrics:
- Posterior samples show clear peaks (not flat distributions)
- 95% credible intervals are narrower than current broad ranges
- Predictive plots show good agreement with observed data

### Phase 2: Training Optimization (Medium Priority) 📈

#### Tasks:
5. **Extend training duration**
   - Increase `max_epochs` from 200 to 500
   - Adjust early stopping criteria for better convergence
   - Monitor training curves for optimal stopping point

6. **Improve training hyperparameters**
   - Test different learning rates (current: 5e-4)
   - Experiment with network architecture (hidden features, transforms)
   - Add learning rate scheduling if needed

7. **Enhance validation**
   - Add posterior quality metrics during training
   - Implement cross-validation for robust assessment
   - Monitor parameter recovery on synthetic data

8. **Increase training data**
   - Generate 100k samples if computational resources allow
   - Test if more data improves posterior quality
   - Balance data quantity vs training time

#### Expected Outcomes:
- **Better training convergence** with optimal hyperparameters
- **More robust parameter estimates** with improved uncertainty quantification
- **Systematic validation** ensuring reliable performance

### Phase 3: Advanced Validation & Deployment (Low Priority) 🔬

#### Tasks:
9. **Simulation-based calibration (SBC)**
   - Implement SBC tests for systematic validation
   - Check posterior recovery across parameter space
   - Identify any systematic biases

10. **Benchmark against particle filter**
    - Compare NPE results with particle filter from external reference
    - Validate parameter estimates against published results
    - Assess computational efficiency gains

11. **Production optimization**
    - Optimize inference speed for clinical application
    - Add robust error handling and validation
    - Create user-friendly interfaces

#### Expected Outcomes:
- **Validated NPE model** with proven accuracy
- **Production-ready system** for clinical inference
- **Performance benchmarks** demonstrating advantages over particle filter

## File Modifications Checklist

### Phase 1 Files (COMPLETED ✅):
- [x] `src/TEIRV/teirv_data_generation.py`
  - [x] Change default `t_max` from 14.0 to 10.0
  - [x] Update time grid creation
  - [x] Modify observation model application

- [x] `scripts/TEIRV_workflow.py`
  - [x] Update workflow default parameters
  - [x] Modify predictive plot time ranges (lines 379-541)
  - [ ] Add training diagnostics *(Phase 2)*

- [x] `scripts/slurm_scripts/run_teirv_workflow.sh`
  - [x] Update SLURM parameters for new training requirements
  - [x] Adjust time limits and resource allocation

### Secondary Files:
- [ ] `src/TEIRV/teirv_inference.py`
  - [ ] Add posterior quality metrics
  - [ ] Enhance training monitoring

- [ ] `src/TEIRV/teirv_utils.py`
  - [ ] Add new utility functions for validation
  - [ ] Enhance plotting capabilities

## Progress Log

### July 4, 2025 - Initial Planning
- ✅ Completed comprehensive code analysis
- ✅ Verified implementation against original paper  
- ✅ Identified root cause: training timespan mismatch
- ✅ Created improvement plan with phased approach
- ✅ **Phase 1 Implementation COMPLETED**

### July 4, 2025 - Phase 1 Implementation ✅
**Core Training Data Fix - COMPLETED**

#### Changes Made:
1. ✅ **Modified data generation timespan**
   - Changed default `t_max` from 14.0 → 10.0 days in `src/TEIRV/teirv_data_generation.py:21`
   - Updated time grid utility in `src/TEIRV/teirv_utils.py:223`
   - Training data now uses 11 points (0-10 days) instead of 15 points (0-14 days)

2. ✅ **Updated workflow parameters**
   - Modified workflow defaults in `scripts/TEIRV_workflow.py:751,789`
   - Updated demo configuration (line 604)
   - Changed clinical inference to use 10-day time grid (line 266)

3. ✅ **Transformed predictive plotting**
   - Modified time grids: `t_obs` = 0-10 days, `t_pred` = 0-20 days (line 409-410)
   - Updated array dimensions and boundary transition (line 453, 504-505)
   - Changed transition marker from day 14 → day 10 (line 519)
   - Updated labels to "Training/Prediction boundary"

4. ✅ **Updated SLURM configuration**
   - Added descriptive comments about 10-day training in `scripts/slurm_scripts/run_teirv_workflow.sh`
   - Clarified workflow description for 10-20 day predictions

#### Expected Impact:
- **Training focus**: Model now learns early infection dynamics (0-10 days)
- **True prediction**: Days 10-20 represent genuine out-of-sample forecasting
- **Better identifiability**: Critical growth phase should provide more informative likelihood
- **Posterior concentration**: Expected narrower, more peaked parameter distributions

### July 4, 2025 - Phase 2 Implementation 📈
**Training Optimization - IN PROGRESS**

#### Rationale:
Phase 1 changes (10-day training) are relatively minor. The **core issue** is likely insufficient training:
- Only **38 epochs** with early stopping (validation loss 17.2066)
- Limited training duration may prevent posterior concentration
- Need longer training + better hyperparameters for complex 6-parameter problem

#### Phase 2 Priority Tasks:
1. **🎯 Extend training duration** (200 → 500 epochs)
2. **⚡ Improve early stopping criteria** (more patience for convergence)
3. **📊 Add training diagnostics** (loss monitoring, posterior quality tracking)
4. **🔧 Optimize hyperparameters** (learning rate, network architecture)
5. **📈 Increase training data** (50k → 100k samples if needed)

#### Changes Made:
1. ✅ **Extended training duration**
   - Increased `max_epochs` from 150/200 → 500 in `src/TEIRV/teirv_inference.py:86`
   - Extended early stopping patience from 25 → 50 epochs
   - Updated workflow and SLURM defaults to match

2. ✅ **Enhanced training diagnostics**
   - Added detailed training configuration reporting (lines 132-137)
   - Added final training results summary (lines 155-158)
   - Created `assess_posterior_quality()` method for quality monitoring

3. ✅ **Increased training data**
   - Raised default training samples from 50k → 100k
   - Updated SLURM script: `N_SAMPLES=100000`
   - Updated workflow default: `n_samples=100000`

4. 🔄 **Network architecture optimization** (pending)
   - Consider larger hidden features if needed
   - Test learning rate scheduling
   - Experiment with different network architectures

#### Expected Impact:
- **500 epochs** should allow much better convergence than previous 38 epochs
- **100k samples** provides more diverse training data for complex 6-parameter space
- **Enhanced monitoring** will show us exactly how training progresses
- **Increased patience** prevents premature stopping on complex problems

#### Next Actions:
- Test new configuration with extended training
- Monitor training curves and convergence behavior
- Compare posterior concentration against 38-epoch baseline
- Adjust network architecture if validation loss plateaus

---

## Technical Notes

### Current Training Configuration (Phase 2 Optimized)
```bash
# From run_teirv_workflow.sh
N_SAMPLES=100000             # Doubled training data
MAX_EPOCHS=500               # Extended for better convergence  
EARLY_STOPPING=50            # Increased patience
HIDDEN_FEATURES=256          # Network architecture
NUM_TRANSFORMS=8             # Normalizing flow complexity
TRAIN_BATCH_SIZE=512         # Training batch size
LEARNING_RATE=5e-4           # Learning rate
```

### Recent Successful Run (Job 1992097)
- **Data Generation**: 50k samples, 14-day trajectories
- **Training**: 38 epochs, validation loss 17.2066, early stopping
- **Inference**: 6 patients, device compatibility working
- **Issue**: Broad posteriors, poor predictive agreement

### Key Performance Metrics
- **Data Generation Rate**: ~2.7 samples/second  
- **Training Time**: ~7 minutes for 38 epochs
- **Inference Time**: <1 second per patient
- **Predictive Plot Generation**: ~90 seconds per patient (20 samples)

## References

1. **Original Paper**: Jump-Switch-Flow for Non-Markovian stochastic PDMP models (Germano et al., 2024)
2. **External Reference**: `external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/`
3. **Config File**: `external/.../config/cli-refractory-tiv-jsf.toml`
4. **Implementation**: `src/TEIRV/` modules

---

*This document will be updated throughout the improvement process to track progress and results.*