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
# 1. Generate training data using unified config
# 2. Train NPE model on GPU
# 3. Run inference on clinical patient data
#
# Usage: sbatch scripts/slurm/run_teirv_workflow.sh <path_to_config.yaml>
#
# All parameters are defined in the provided config file

# Check if a config file is provided
if [ -z "$1" ]; then
  echo "Error: No config file provided."
  echo "Usage: sbatch scripts/slurm/run_teirv_workflow.sh <path_to_config.yaml>"
  exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file '$CONFIG_FILE' not found."
  exit 1
fi

# Workflow name (only parameter not in config)
WORKFLOW_NAME="production_run_$(date +%Y%m%d_%H%M%S)"

# Extract key parameters from config for display (using Python)
eval $(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(f'N_SAMPLES={config[\"data\"][\"generation\"][\"n_samples\"]}')
print(f'DEVICE={config[\"system\"][\"device\"]}')
print(f'MAX_EPOCHS={config[\"training\"][\"max_num_epochs\"]}')
print(f'HIDDEN_FEATURES={config[\"network\"][\"hidden_features\"]}')
print(f'NUM_TRANSFORMS={config[\"network\"][\"num_transforms\"]}')
")

echo "=========================================="
echo "TEIRV NPE Full Workflow"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Config file: $CONFIG_FILE"
echo "Config file exists: $([ -f "$CONFIG_FILE" ] && echo "✅ Yes" || echo "❌ No")"
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

# Run the complete TEIRV workflow using config file
echo "Starting TEIRV workflow at $(date)"

echo "Command: python -u src/main.py full --config $CONFIG_FILE --workflow_name $WORKFLOW_NAME"

time python -u src/main.py full --config $CONFIG_FILE --workflow_name "$WORKFLOW_NAME"

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