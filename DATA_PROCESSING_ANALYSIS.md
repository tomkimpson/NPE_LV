# Patient 432192 Data Discrepancy Analysis

## Problem Statement
User noticed that patient 432192 appears to have two identical initial viral load values in our analysis, one at t=0 and one later, but the original JSF data only shows a single value at t=1.

## Root Cause Analysis

### Original Data Sources
1. **JSF Germano2024 .ssv files**: Start at t=1 day
   - `external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/432192.ssv`
   - First observation: t=1, log10V=4.645
   
2. **Processed CSV files**: Start at t=0 day  
   - `external/JSFGermano2024/.../processing/COVID_Results/PatientData/CSVs/432192.csv`
   - First observation: t=0, log10V=4.645

### NPE_LV Data Processing Pipeline

#### Step 1: Clinical Data Loading
- **File**: `src/TEIRV/clinical_data.py`
- **Function**: `load_patient_data()` (line 49)
- **Source**: Loads from `.ssv` files (t=1 start)
- **Result**: times=[1,2,3,...,14], observations=[4.645, 6.157, ...]

#### Step 2: Time Grid Creation  
- **File**: `scripts/TEIRV_workflow.py` (line 266)
- **Function**: `create_teirv_time_grid(10.0, 1.0)`
- **Result**: target_times=[0,1,2,3,4,5,6,7,8,9,10]

#### Step 3: Interpolation
- **File**: `src/TEIRV/clinical_data.py` (line 168)
- **Function**: `np.interp(target_times, times, observations)`
- **Behavior**: Constant extrapolation for points outside range
- **Result**: t=0 gets same value as t=1 (4.645)

## Technical Explanation

```python
# Original patient data (from .ssv)
times = [1, 2, 3, 4, ...]
observations = [4.645, 6.157, 7.727, 6.725, ...]

# Target time grid (for NPE training)
target_times = [0, 1, 2, 3, 4, ...]

# NumPy interpolation with extrapolation
interpolated = np.interp(target_times, times, observations)
# Result: [4.645, 4.645, 6.157, 7.727, 6.725, ...]
#          ^t=0   ^t=1   ^t=2    ^t=3    ^t=4
```

## Implications

### 1. **Modeling Assumption**
- Our NPE model assumes viral load at t=0 (symptom onset) equals the first measurement at t=1
- This is a **constant extrapolation assumption** rather than a biological model

### 2. **Parameter Inference Impact**
- **V₀ parameter**: Inferred initial viral load is based on interpolated t=0 value
- **Early dynamics**: Model fits to artificially extended early trajectory
- **Comparison validity**: Affects comparison with JSF Germano2024 results

### 3. **Biological Interpretation**
- **Original JSF**: First measurement 1 day post-symptom onset
- **Our model**: Assumes same viral load at symptom onset (t=0)
- **Assumption validity**: May not reflect true viral kinetics

## Consistency Across Patients

This extrapolation behavior affects **all patients** consistently:
- All original .ssv files start at t=1
- All get interpolated to include t=0 with same value as t=1
- Processing is uniform across the dataset

## Recommendations

### 1. **Data Processing Options**
- **Option A**: Start time grid at t=1 to match original data
- **Option B**: Use biological model for t=0 extrapolation
- **Option C**: Document current assumption clearly

### 2. **Parameter Interpretation**
- V₀ represents viral load at symptom onset (t=0) via extrapolation
- Early dynamics parameters may be influenced by this assumption
- Comparisons with JSF should account for different time alignment

### 3. **Future Improvements**
- Consider biological modeling for pre-symptom viral dynamics
- Evaluate sensitivity to time alignment assumptions
- Document preprocessing decisions explicitly

## Files Involved

### Data Loading
- `src/TEIRV/clinical_data.py`: Clinical data loader and preprocessing
- `src/TEIRV/teirv_utils.py`: Time grid creation and utilities

### Workflow
- `scripts/TEIRV_workflow.py`: Main inference pipeline
- `workflows/create_predictive_plots.py`: Plotting and visualization

### Data Sources
- `external/JSFGermano2024/TEIVR_Results/particle-filter-example-tiv_covid/data/*.ssv`: Original clinical data
- `workflows/production_run_*/inference_results/patient_*/observations.npy`: Processed observations