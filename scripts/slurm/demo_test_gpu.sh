#!/bin/bash 

#SBATCH --job-name=teirv_demo_test
#SBATCH --output=results/experiments/slurm_outputs/logfiles/teirv_demo_test_%j.out
#SBATCH --error=results/experiments/slurm_outputs/logfiles/teirv_demo_test_%j.err
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00 
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# TEIRV NPE Demo Test SLURM Script
# 
# This script runs a quick demo test of the TEIRV NPE system to validate:
# 1. The new src/main.py interface works correctly in slurm
# 2. GPU acceleration is working properly
# 3. All imports and dependencies are functioning
# 4. End-to-end demo workflow completes successfully
#
# The demo includes:
# - Generate small training dataset (1000 samples)
# - Train basic NPE model on GPU
# - Run inference on clinical patient data
#
# Usage: sbatch scripts/slurm/demo_test_gpu.sh
#
# Expected runtime: 5-10 minutes

# Demo parameters (minimal for quick testing)
WORKFLOW_NAME="demo_test_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"                # Use GPU for testing

echo "=========================================="
echo "TEIRV NPE Demo Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Workflow name: $WORKFLOW_NAME"
echo "Device: $DEVICE"
echo "Test purpose: Validate new src/main.py interface"
echo "=========================================="

# Create output directories
mkdir -p results/experiments/slurm_outputs/logfiles
mkdir -p results/experiments/production_runs

# Activate conda environment
source ~/.bashrc
conda activate NPE_LV

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "NVIDIA GPU info:"
nvidia-smi
echo "=========================================="

# Quick import test to catch issues early
echo "Testing Python imports..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from teirv_data_generation import TEIRVDataGenerator
    from teirv_inference import TEIRVInference
    from clinical_data import ClinicalStudy
    from teirv_utils import create_teirv_prior
    from teirv_simulator import gillespie_teirv
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

IMPORT_EXIT_CODE=$?
if [ $IMPORT_EXIT_CODE -ne 0 ]; then
    echo "❌ IMPORT TEST FAILED - Aborting demo"
    exit 1
fi

echo "=========================================="

# Change to project directory
cd /fred/oz022/tkimpson/NPE_LV

# Run the TEIRV demo
echo "Starting TEIRV demo test at $(date)"

time python -u src/main.py demo

DEMO_EXIT_CODE=$?

echo "=========================================="
echo "Demo test completed at $(date)"
echo "Exit code: $DEMO_EXIT_CODE"

if [ $DEMO_EXIT_CODE -eq 0 ]; then
    echo "✅ DEMO TEST COMPLETED SUCCESSFULLY"
    echo "📁 Results available in: results/experiments/production_runs/demo"
    echo ""
    echo "Generated files:"
    echo "  - results/experiments/production_runs/demo/demo_data.pkl"
    echo "  - results/experiments/production_runs/demo/demo_model.pkl" 
    echo "  - results/experiments/production_runs/demo/inference_results/"
    echo ""
    
    # Quick validation of demo results
    echo "🔍 Demo Results Validation:"
    echo "----------------------------------------"
    
    # Check if key demo files exist
    if [ -f "results/experiments/production_runs/demo/demo_data.pkl" ]; then
        echo "✅ Demo training data generated successfully"
    else
        echo "❌ Demo training data missing"
    fi
    
    if [ -f "results/experiments/production_runs/demo/demo_model.pkl" ]; then
        echo "✅ Demo NPE model trained successfully"
    else
        echo "❌ Demo NPE model missing"
    fi
    
    if [ -f "results/experiments/production_runs/demo/inference_results/clinical_parameter_estimates.csv" ]; then
        echo "✅ Demo clinical inference completed successfully"
        echo "Number of patients analyzed:"
        wc -l < results/experiments/production_runs/demo/inference_results/clinical_parameter_estimates.csv
    else
        echo "❌ Demo clinical inference missing"
    fi
    
    echo ""
    echo "To view detailed results:"
    echo "  ls -la results/experiments/production_runs/demo/"
    echo "  cat results/experiments/production_runs/demo/inference_results/clinical_parameter_estimates.csv"
    
else
    echo "❌ DEMO TEST FAILED with exit code $DEMO_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - Standard output: results/experiments/slurm_outputs/logfiles/teirv_demo_test_${SLURM_JOB_ID}.out"
    echo "  - Standard error: results/experiments/slurm_outputs/logfiles/teirv_demo_test_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues to check:"
    echo "  - Conda environment activation"
    echo "  - CUDA/GPU availability"
    echo "  - Python import paths"
    echo "  - Clinical data availability"
fi

echo "=========================================="

# Print final summary
echo "Demo Test Summary:"
echo "- Job ID: $SLURM_JOB_ID"
echo "- Node: $SLURMD_NODENAME"
echo "- Start: $(date)"
echo "- Device: $DEVICE"
echo "- Exit Code: $DEMO_EXIT_CODE"
echo "- Status: $([ $DEMO_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "=========================================="