#!/bin/bash
#SBATCH --job-name=particle_filter
#SBATCH --output=reproduce_particle_filter_results/logs/particle_filter_%j.out
#SBATCH --error=reproduce_particle_filter_results/logs/particle_filter_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Particle Filter SLURM Submission Script
#
# This script runs particle filter inference using JSF simulation for TEIRV
# viral dynamics parameter estimation on COVID-19 patient data.
#
# Based on: Germano et al. (2024) "Jump-Switch-Flow for Non-Markovian stochastic PDMP models"
#
# Usage:
#   sbatch run_particle_filter.sh                    # Run with defaults
#   sbatch --export=PATIENT_ID=443108 run_particle_filter.sh  # Different patient
#   sbatch --export=N_PARTICLES=3000 run_particle_filter.sh   # Fewer particles

# =============================================================================
# Configuration
# =============================================================================

# Patient ID (default: 432192)
# Available patients: 432192, 443108, 444332, 444391, 445602, 451152
PATIENT_ID=${PATIENT_ID:-432192}

# Number of particles (default: 6000)
# For quick tests, use 100-1000 (not recommended for actual inference)
N_PARTICLES=${N_PARTICLES:-6000}

# Output directory (default: results)
OUTPUT_DIR=${OUTPUT_DIR:-results}

# =============================================================================
# Job Information
# =============================================================================

echo "=========================================="
echo "Particle Filter Inference - TEIRV Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Patient ID: $PATIENT_ID"
echo "  Particles: $N_PARTICLES"
echo "  Output: $OUTPUT_DIR/patient_$PATIENT_ID/"
echo ""
echo "Method: Sequential Monte Carlo (Particle Filter)"
echo "Simulator: Jump-Switch-Flow (JSF)"
echo "Model: TEIRV (5 compartments, 7 reactions)"
echo "=========================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment
echo ""
echo "Activating conda environment..."
source ~/.bashrc
conda activate NPE_LV

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import pypfilt; print(f'pypfilt version: {pypfilt.__version__}')" || echo "⚠️  pypfilt not found"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')" || echo "⚠️  numpy not found"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')" || echo "⚠️  pandas not found"
python -c "import matplotlib; print(f'matplotlib version: {matplotlib.__version__}')" || echo "⚠️  matplotlib not found"
python -c "import scipy; print(f'scipy version: {scipy.__version__}')" || echo "⚠️  scipy not found"
python -c "import tqdm; print(f'tqdm version: {tqdm.__version__}')" || echo "⚠️  tqdm not found"
echo "=========================================="

# =============================================================================
# Change to project directory
# =============================================================================

cd /fred/oz022/tkimpson/NPE_LV
echo ""
echo "Working directory: $(pwd)"
echo "=========================================="

# =============================================================================
# Runtime Estimate
# =============================================================================

echo ""
echo "⚠️  COMPUTATIONAL COST WARNING ⚠️"
echo ""

# Typical number of observation time points
N_OBS=14

# Estimate total JSF simulations
N_SIMS=$((N_PARTICLES * N_OBS))

echo "Estimated JSF simulations: ~$N_SIMS"
echo ""

if [ $N_PARTICLES -ge 6000 ]; then
    echo "Expected runtime: 1-6 hours (highly variable)"
    echo "Recommendation: Production run (6000 particles)"
elif [ $N_PARTICLES -ge 3000 ]; then
    echo "Expected runtime: 30 minutes - 3 hours"
    echo "Recommendation: Medium-quality inference"
elif [ $N_PARTICLES -ge 1000 ]; then
    echo "Expected runtime: 10-60 minutes"
    echo "Recommendation: Acceptable for testing"
else
    echo "Expected runtime: 1-10 minutes"
    echo "⚠️  WARNING: <1000 particles NOT recommended for real inference!"
    echo "   Use only for code testing/debugging"
fi

echo ""
echo "Resources:"
echo "  CPUs: 1 (single-threaded JSF)"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo "  Time limit: 12 hours"
echo ""
echo "Progress updates will appear in real-time below."
echo "=========================================="

# =============================================================================
# Run Particle Filter
# =============================================================================

echo ""
echo "Starting particle filter inference at $(date)"
echo "=========================================="
echo ""

# Run with time tracking
time python -u reproduce_particle_filter_results/particle_filter.py \
    --patient $PATIENT_ID \
    --particles $N_PARTICLES \
    --output $OUTPUT_DIR \
    --verbose

EXIT_CODE=$?

# =============================================================================
# Post-Processing
# =============================================================================

echo ""
echo "=========================================="
echo "Particle filter completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ INFERENCE COMPLETED SUCCESSFULLY"
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR/patient_$PATIENT_ID/"
    echo ""

    # Check for output files
    RESULT_DIR="$OUTPUT_DIR/patient_$PATIENT_ID"

    if [ -d "$RESULT_DIR" ]; then
        echo "Generated files:"
        echo "----------------------------------------"
        ls -lh "$RESULT_DIR"
        echo "----------------------------------------"
        echo ""

        # Check key files
        if [ -f "$RESULT_DIR/fit_result.pkl" ]; then
            echo "✅ Particle filter results: fit_result.pkl"
        else
            echo "⚠️  Missing: fit_result.pkl"
        fi

        if [ -f "$RESULT_DIR/beta_samples.csv" ]; then
            echo "✅ Posterior samples: *_samples.csv"
        else
            echo "⚠️  Missing: parameter samples"
        fi

        if [ -f "$RESULT_DIR/patient_${PATIENT_ID}_posteriors.png" ]; then
            echo "✅ Posterior plots: patient_${PATIENT_ID}_*.png"
        else
            echo "⚠️  Missing: visualization plots"
        fi

        echo ""
        echo "📊 To load results in Python:"
        echo "  import pickle"
        echo "  with open('$RESULT_DIR/fit_result.pkl', 'rb') as f:"
        echo "      results = pickle.load(f)"
        echo ""

        # Show file sizes
        TOTAL_SIZE=$(du -sh "$RESULT_DIR" | cut -f1)
        echo "Total output size: $TOTAL_SIZE"

    else
        echo "⚠️  Output directory not found: $RESULT_DIR"
    fi

    echo ""
    echo "🎯 Next Steps:"
    echo "  1. Compare with NPE results from src/teirv_inference.py"
    echo "  2. Validate on other patients (443108, 444332, etc.)"
    echo "  3. Analyze parameter posteriors for identifiability"
    echo ""

else
    echo ""
    echo "❌ INFERENCE FAILED"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Check error log for details:"
    echo "  reproduce_particle_filter_results/logs/particle_filter_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  - Patient data not found (check PATIENT_ID)"
    echo "  - Memory exceeded (increase --mem)"
    echo "  - Time limit exceeded (increase --time)"
    echo "  - Conda environment not activated"
    echo ""
fi

echo "=========================================="

# =============================================================================
# Job Summary
# =============================================================================

echo ""
echo "📋 JOB SUMMARY:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Patient: $PATIENT_ID"
echo "  Particles: $N_PARTICLES"
echo "  Status: $([ $EXIT_CODE -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "  Runtime: See above"
echo "=========================================="
echo ""

exit $EXIT_CODE
