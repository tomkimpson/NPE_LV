#!/bin/bash 

# Check if a config file is provided (needed for SBATCH parameters)
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

# Validate that the config file contains the required slurm section
if ! yq ".slurm" "$CONFIG_FILE" > /dev/null 2>&1; then
  echo "Error: Config file '$CONFIG_FILE' is missing the required 'slurm' section."
  echo "Please add a 'slurm' section with SBATCH parameters to your config file."
  exit 1
fi

# Validate required slurm parameters
required_params=("job_name" "output" "error" "time" "mem" "cpus_per_task")
for param in "${required_params[@]}"; do
  if [ "$(yq ".slurm.$param" "$CONFIG_FILE")" == "null" ]; then
    echo "Error: Missing required slurm parameter: $param"
    exit 1
  fi
done

# Dynamic SBATCH parameters from config file
#SBATCH --job-name=$(yq ".slurm.job_name" "$CONFIG_FILE")
#SBATCH --output=$(yq ".slurm.output" "$CONFIG_FILE")
#SBATCH --error=$(yq ".slurm.error" "$CONFIG_FILE")
#SBATCH --export=$(yq ".slurm.export" "$CONFIG_FILE")
#SBATCH --gres=$(yq ".slurm.gres" "$CONFIG_FILE")
#SBATCH --time=$(yq ".slurm.time" "$CONFIG_FILE")
#SBATCH --mem=$(yq ".slurm.mem" "$CONFIG_FILE")
#SBATCH --cpus-per-task=$(yq ".slurm.cpus_per_task" "$CONFIG_FILE")

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



# Extract key parameters from config for display (using Python)
eval $(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
print(f'JOB_NAME={config[\"slurm\"][\"job_name\"]}')
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
echo "Job name: $JOB_NAME"
echo "Training samples: $N_SAMPLES"
echo "Device: $DEVICE"
echo "Max epochs: $MAX_EPOCHS"
echo "Hidden features: $HIDDEN_FEATURES"
echo "Transforms: $NUM_TRANSFORMS"
echo "=========================================="

# Create output directories
mkdir -p results/experiments/slurm_outputs/logfiles

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

echo "Command: python -u src/main.py full --config $CONFIG_FILE"

time python -u src/main.py full --config $CONFIG_FILE

WORKFLOW_EXIT_CODE=$?

echo "=========================================="
echo "Workflow completed at $(date)"
echo "Exit code: $WORKFLOW_EXIT_CODE"

if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    echo "✅ WORKFLOW COMPLETED SUCCESSFULLY"
    echo "📁 Results available in standard output directories"
    echo ""
    echo "Generated files:"
    echo "  - models/production/npe_model.pkl" 
    echo "  - results/predictions/inference_results/"
    echo ""
    echo "To view results:"
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
#     cp -r results/predictions/inference_results /path/to/backup/location/
# fi