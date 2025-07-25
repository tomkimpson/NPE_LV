#!/bin/bash 

#SBATCH --job-name=teirv_test
#SBATCH --output=outputs/logfiles/teirv_test_%j.out
#SBATCH --error=outputs/logfiles/teirv_test_%j.err
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# TEIRV NPE Test Workflow SLURM Script
# 
# Lightweight test to verify the updated TEIRV parameter scaling works correctly.
# This script runs a minimal version of the full pipeline to:
# 1. Generate small training dataset (5,000 samples)
# 2. Train a simple NPE model 
# 3. Run inference and verify posteriors respect the new scaling
#
# Usage: sbatch scripts/slurm_scripts/test_teirv_workflow.sh

# Test workflow parameters (reduced for speed)
N_SAMPLES=5000               # Small dataset for quick test
WORKFLOW_NAME="test_scaling_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"                # Use GPU for faster training
MAX_EPOCHS=200               # Reduced epochs for test
HIDDEN_FEATURES=128          # Smaller network for test
NUM_TRANSFORMS=4             # Fewer transforms for test

# Training parameters (reduced)
TRAIN_BATCH_SIZE=256         # Smaller batch size
LEARNING_RATE=1e-3           # Slightly higher LR for faster convergence
INFERENCE_SAMPLES=2000       # Fewer posterior samples

# Clinical filtering (relaxed for test)
MIN_DETECTIONS=3             # Lower threshold for test
MIN_PEAK_VL=1.0              # Lower threshold for test

echo "=========================================="
echo "TEIRV NPE Parameter Scaling Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Workflow name: $WORKFLOW_NAME"
echo "Training samples: $N_SAMPLES"
echo "Device: $DEVICE"
echo "Max epochs: $MAX_EPOCHS"
echo "Hidden features: $HIDDEN_FEATURES"
echo "Transforms: $NUM_TRANSFORMS"
echo ""
echo "🎯 Test Goals:"
echo "  - Verify parameter scaling works correctly"
echo "  - Check that β×10^-7 and φ×10^-5 specifications are respected"
echo "  - Ensure prior bounds: β(0,20), π(200,400), δ(1,10), φ(0,15), ρ(0,1)"
echo "  - Confirm V0 log-scaling: V0 ~ exp(Uniform(0,5))"
echo "=========================================="

# Create output directories
mkdir -p outputs/logfiles
mkdir -p workflows

# Activate conda environment
source ~/.bashrc
conda activate NPE_LV

# Check CUDA availability if using GPU
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking CUDA availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    echo "NVIDIA GPU info:"
    nvidia-smi | head -15
    echo "=========================================="
fi

# Change to project directory
cd /fred/oz022/tkimpson/NPE_LV

# First, run the parameter scaling verification test
echo "🧪 Running parameter scaling verification test..."
echo "=========================================="
python test_parameter_scaling_equivalence.py
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Parameter scaling test FAILED - aborting workflow"
    exit 1
fi

echo "✅ Parameter scaling test PASSED - proceeding with workflow"
echo "=========================================="

# Run the lightweight TEIRV workflow
echo "Starting test TEIRV workflow at $(date)"

time python -u scripts/TEIRV_workflow.py \
    --device $DEVICE \
    full \
    --workflow_name "$WORKFLOW_NAME" \
    --n_samples $N_SAMPLES \
    --max_epochs $MAX_EPOCHS \
    --hidden_features $HIDDEN_FEATURES \
    --num_transforms $NUM_TRANSFORMS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --inference_samples $INFERENCE_SAMPLES \
    --min_detections $MIN_DETECTIONS \
    --min_peak_vl $MIN_PEAK_VL

WORKFLOW_EXIT_CODE=$?

echo "=========================================="
echo "Test workflow completed at $(date)"
echo "Exit code: $WORKFLOW_EXIT_CODE"

if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    echo "✅ TEST WORKFLOW COMPLETED SUCCESSFULLY"
    echo "📁 Results available in: workflows/$WORKFLOW_NAME"
    echo ""
    
    # Quick validation of results
    echo "🔍 Quick Results Validation:"
    echo "----------------------------------------"
    
    # Check if key files exist
    if [ -f "workflows/$WORKFLOW_NAME/training_data.pkl" ]; then
        echo "✅ Training data generated successfully"
    else
        echo "❌ Training data missing"
    fi
    
    if [ -f "workflows/$WORKFLOW_NAME/npe_model.pkl" ]; then
        echo "✅ NPE model trained successfully"
    else
        echo "❌ NPE model missing"
    fi
    
    if [ -f "workflows/$WORKFLOW_NAME/inference_results/clinical_parameter_estimates.csv" ]; then
        echo "✅ Clinical inference completed successfully"
        echo ""
        echo "📊 Parameter Estimates Preview:"
        echo "----------------------------------------"
        head -3 "workflows/$WORKFLOW_NAME/inference_results/clinical_parameter_estimates.csv"
        echo "----------------------------------------"
        echo "ℹ️  Parameter interpretation:"
        echo "  - β_mean: infection rate (should be ≈ raw_β × 10^-7)"
        echo "  - φ_mean: interferon protection (should be ≈ raw_φ × 10^-5)"
        echo "  - π_mean: virion production (200-400 range)"
        echo "  - δ_mean: cell clearance (1-10 range)"
        echo "  - V₀_mean: initial virions (exp-distributed)"
    else
        echo "❌ Clinical inference missing"
    fi
    
    echo ""
    echo "Generated files:"
    ls -la "workflows/$WORKFLOW_NAME/"
    
    echo ""
    echo "🎉 TEST COMPLETED - Parameter scaling verification successful!"
    echo "Ready for production runs with full parameters."
    
else
    echo "❌ TEST WORKFLOW FAILED with exit code $WORKFLOW_EXIT_CODE"
    echo "🚨 Parameter scaling may have issues - check logs before production"
    echo ""
    echo "Check the log files for details:"
    echo "  - Standard output: outputs/logfiles/teirv_test_${SLURM_JOB_ID}.out"
    echo "  - Standard error: outputs/logfiles/teirv_test_${SLURM_JOB_ID}.err"
fi

echo "=========================================="

# Final summary
echo "📋 TEST SUMMARY:"
echo "  Parameter scaling test: ${TEST_EXIT_CODE:-FAILED}"
echo "  Full workflow test: $WORKFLOW_EXIT_CODE"
echo "  Overall status: $([ $WORKFLOW_EXIT_CODE -eq 0 ] && echo "✅ READY FOR PRODUCTION" || echo "❌ NEEDS DEBUGGING")"
echo "=========================================="