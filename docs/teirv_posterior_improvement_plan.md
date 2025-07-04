# TEIRV NPE Posterior Quality Improvement Plan

**Date Created:** July 4, 2025  
**Branch:** teirv-implementation  
**Status:** Planning Phase

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

### Primary Files to Modify:
- [ ] `src/TEIRV/teirv_data_generation.py`
  - [ ] Change default `t_max` from 14.0 to 10.0
  - [ ] Update time grid creation
  - [ ] Modify observation model application

- [ ] `scripts/TEIRV_workflow.py`
  - [ ] Update workflow default parameters
  - [ ] Modify predictive plot time ranges (lines 379-541)
  - [ ] Add training diagnostics

- [ ] `scripts/slurm_scripts/run_teirv_workflow.sh`
  - [ ] Update SLURM parameters for new training requirements
  - [ ] Adjust time limits and resource allocation

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
- 📝 Next: Begin Phase 1 implementation

### Future Updates
*This section will be updated as implementation progresses*

---

## Technical Notes

### Current Training Configuration
```bash
# From run_teirv_workflow.sh
N_SAMPLES=50000
MAX_EPOCHS=200
HIDDEN_FEATURES=256
NUM_TRANSFORMS=8
TRAIN_BATCH_SIZE=512
LEARNING_RATE=5e-4
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