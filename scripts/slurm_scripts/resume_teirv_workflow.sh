#!/bin/bash 

#SBATCH --job-name=teirv_workflow_resume
#SBATCH --output=outputs/logfiles/teirv_workflow_resume_%j.out
#SBATCH --error=outputs/logfiles/teirv_workflow_resume_%j.err
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# TEIRV NPE Resume Workflow SLURM Script
# 
# This script resumes the TEIRV NPE pipeline by:
# 1. Using existing training data from production_run_20250709_144328
# 2. Training NPE model on GPU (skipping data generation)
# 3. Running inference on clinical patient data with 10-20 day predictions
#
# Usage: sbatch scripts/slurm_scripts/resume_teirv_workflow.sh

# Resume parameters
EXISTING_DATA="workflows/production_run_20250709_144328/training_data.pkl"
WORKFLOW_NAME="resume_run_$(date +%Y%m%d_%H%M%S)"
DEVICE="cuda"                # Use GPU for training
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
echo "TEIRV NPE Resume Workflow"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Workflow name: $WORKFLOW_NAME"
echo "Using existing data: $EXISTING_DATA"
echo "Device: $DEVICE"
echo "Max epochs: $MAX_EPOCHS"
echo "Hidden features: $HIDDEN_FEATURES"
echo "Transforms: $NUM_TRANSFORMS"
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
    nvidia-smi
    echo "=========================================="
fi

# Change to project directory
cd /fred/oz022/tkimpson/NPE_LV

# Check if existing training data exists
if [ ! -f "$EXISTING_DATA" ]; then
    echo "❌ ERROR: Training data not found at $EXISTING_DATA"
    echo "Please run the full workflow first or check the path."
    exit 1
fi

echo "✅ Found existing training data: $EXISTING_DATA"
echo "📊 Training data size:"
python -c "
import pickle
with open('$EXISTING_DATA', 'rb') as f:
    data = pickle.load(f)
    print(f'  - Parameter samples: {data[\"theta\"].shape}')
    print(f'  - Observations: {data[\"x\"].shape}')
    print(f'  - Observation type: {data[\"observation_type\"]}')
"

# Create workflow directory
mkdir -p workflows/$WORKFLOW_NAME

# Step 1: Train NPE model using existing data
echo "🧠 Step 1: Training NPE model at $(date)"

time python -u scripts/TEIRV_workflow.py \
    --device $DEVICE \
    train \
    --data "$EXISTING_DATA" \
    --output "workflows/$WORKFLOW_NAME/npe_model.pkl" \
    --max_epochs $MAX_EPOCHS \
    --hidden_features $HIDDEN_FEATURES \
    --num_transforms $NUM_TRANSFORMS \
    --batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LEARNING_RATE

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "✅ Training completed successfully"

# Step 2: Run inference on clinical data  
echo "🏥 Step 2: Running inference at $(date)"

time python -u scripts/TEIRV_workflow.py \
    --device $DEVICE \
    inference \
    --model "workflows/$WORKFLOW_NAME/npe_model.pkl" \
    --output "workflows/$WORKFLOW_NAME/inference_results" \
    --n_samples $INFERENCE_SAMPLES \
    --min_detections $MIN_DETECTIONS \
    --min_peak_vl $MIN_PEAK_VL

WORKFLOW_EXIT_CODE=$?

echo "=========================================="
echo "Resume workflow completed at $(date)"
echo "Exit code: $WORKFLOW_EXIT_CODE"

if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    echo "✅ RESUME WORKFLOW COMPLETED SUCCESSFULLY"
    echo "📁 Results available in: workflows/$WORKFLOW_NAME"
    echo ""
    echo "Generated files:"
    echo "  - workflows/$WORKFLOW_NAME/npe_model.pkl" 
    echo "  - workflows/$WORKFLOW_NAME/inference_results/"
    echo ""
    echo "To view results:"
    echo "  ls -la workflows/$WORKFLOW_NAME/"
    echo "  cat workflows/$WORKFLOW_NAME/inference_results/clinical_parameter_estimates.csv"
else
    echo "❌ RESUME WORKFLOW FAILED with exit code $WORKFLOW_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - Standard output: outputs/logfiles/teirv_workflow_resume_${SLURM_JOB_ID}.out"
    echo "  - Standard error: outputs/logfiles/teirv_workflow_resume_${SLURM_JOB_ID}.err"
fi

echo "=========================================="