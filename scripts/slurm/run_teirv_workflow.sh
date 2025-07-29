#!/bin/bash 

#SBATCH --job-name=teirv_workflow_full
#SBATCH --output=results/experiments/slurm_outputs/logfiles/teirv_workflow_%j.out
#SBATCH --error=results/experiments/slurm_outputs/logfiles/teirv_workflow_%j.err
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# TEIRV NPE Full Workflow SLURM Script
# 
# This script runs the complete TEIRV NPE pipeline:
# 1. Generate training data (50,000 samples, 10-day trajectories)
# 2. Train NPE model on GPU
# 3. Run inference on clinical patient data with 10-20 day predictions
#
# Usage: sbatch scripts/slurm_scripts/run_teirv_workflow.sh
#
# Modify the parameters below as needed:

# Workflow parameters
N_SAMPLES=50000              # Reduced training samples for faster completion
WORKFLOW_NAME="production_run_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"                # Use GPU for training
# NOTE: Uses t_max=10.0 days by default (training on early infection dynamics)
MAX_EPOCHS=1000              # Extended training epochs for better convergence
HIDDEN_FEATURES=512          # Larger neural network for better expressivity
NUM_TRANSFORMS=12            # More transforms for better posterior approximation

# Training parameters
TRAIN_BATCH_SIZE=512         # Training batch size
LEARNING_RATE=5e-4           # Learning rate
INFERENCE_SAMPLES=20000      # Posterior samples for inference

# Clinical filtering
MIN_DETECTIONS=5             # Minimum detections for patient inclusion
MIN_PEAK_VL=2.0             # Minimum peak viral load

echo "=========================================="
echo "TEIRV NPE Full Workflow"
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
echo "=========================================="

# Create output directories
mkdir -p results/experiments/slurm_outputs/logfiles
mkdir -p results/experiments/production_runs

# Activate conda environment
source ~/.bashrc
conda activate NPE_LV

# Check CUDA availability if using GPU
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking CUDA availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    echo "NVIDIA GPU info:"
    nvidia-smi
    echo "=========================================="
fi

# Change to project directory
cd /fred/oz022/tkimpson/NPE_LV

# Run the complete TEIRV workflow
echo "Starting TEIRV workflow at $(date)"

time python -u src/main.py \
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
echo "Workflow completed at $(date)"
echo "Exit code: $WORKFLOW_EXIT_CODE"

if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    echo "✅ WORKFLOW COMPLETED SUCCESSFULLY"
    echo "📁 Results available in: results/experiments/production_runs/$WORKFLOW_NAME"
    echo ""
    echo "Generated files:"
    echo "  - results/experiments/production_runs/$WORKFLOW_NAME/training_data.pkl"
    echo "  - models/production/npe_model.pkl" 
    echo "  - results/predictions/inference_results/"
    echo ""
    echo "To view results:"
    echo "  ls -la results/experiments/production_runs/$WORKFLOW_NAME/"
    echo "  cat results/predictions/inference_results/clinical_parameter_estimates.csv"
else
    echo "❌ WORKFLOW FAILED with exit code $WORKFLOW_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - Standard output: results/experiments/slurm_outputs/logfiles/teirv_workflow_${SLURM_JOB_ID}.out"
    echo "  - Standard error: results/experiments/slurm_outputs/logfiles/teirv_workflow_${SLURM_JOB_ID}.err"
fi

echo "=========================================="

# Optionally, copy important results to a backup location
# if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
#     echo "Backing up results..."
#     cp -r results/experiments/production_runs/$WORKFLOW_NAME /path/to/backup/location/
# fi