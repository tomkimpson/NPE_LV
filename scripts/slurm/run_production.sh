#!/bin/bash

#SBATCH --job-name=teirv_production_%j
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# NOTE: Time allocation breakdown for full pipeline with data generation:
# - Data generation (200k samples): ~24 hours
# - Model training: ~1 hour
# - Inference: ~5 minutes
# Total: ~36 hours (set to 36:00:00 for safety margin)

# TEIRV NPE Production Run SLURM Script
#
# This script runs the complete TEIRV NPE pipeline with the new directory structure:
# 1. Handle training data (use existing or generate new based on config)
# 2. Train NPE model on GPU
# 3. Run inference on clinical patient data
#
# All outputs are organized in a single timestamped run directory:
#   results/run_YYYYMMDD_HHMMSS/
#     ├── config.yaml          # Config used for this run
#     ├── data_source.txt      # Which training data was used
#     ├── model/               # Trained model
#     ├── logs/                # All logs (training + slurm)
#     └── inference/           # Patient inference results
#
# Usage: sbatch scripts/slurm/run_production.sh
#
# All settings are defined in configs/teirv_config.yaml

# Config file to use (edit this line to change config)
CONFIG_FILE="configs/teirv_config.yaml"

echo "=========================================="
echo "TEIRV NPE Production Workflow"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
conda activate NPE_LV

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "NVIDIA GPU info:"
nvidia-smi
echo "=========================================="

# Change to project directory
cd /fred/oz022/tkimpson/NPE_LV

# Run the complete TEIRV workflow
echo "Starting TEIRV production workflow at $(date)"
echo "Command: python -u src/main.py full --config $CONFIG_FILE"
echo ""

time python -u src/main.py full --config "$CONFIG_FILE"

WORKFLOW_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Workflow completed at $(date)"
echo "Exit code: $WORKFLOW_EXIT_CODE"

# Move SLURM output file to results directory
SLURM_OUTPUT="slurm-${SLURM_JOB_ID}.out"
RESULTS_DIR=$(ls -td results/run_* 2>/dev/null | head -1)

if [ -n "$RESULTS_DIR" ] && [ -d "$RESULTS_DIR" ]; then
    LOGS_DIR="${RESULTS_DIR}/logs"
    if [ -d "$LOGS_DIR" ]; then
        echo "Moving SLURM output to ${LOGS_DIR}/${SLURM_OUTPUT}"
        mv "$SLURM_OUTPUT" "${LOGS_DIR}/" 2>/dev/null || echo "Warning: Could not move SLURM output file"
    fi
fi

if [ $WORKFLOW_EXIT_CODE -eq 0 ]; then
    echo "✅ PRODUCTION WORKFLOW COMPLETED SUCCESSFULLY"
    echo ""
    echo "All results are in: $RESULTS_DIR"
    echo "SLURM logs moved to: ${LOGS_DIR}/${SLURM_OUTPUT}"
    echo ""
else
    echo "❌ WORKFLOW FAILED with exit code $WORKFLOW_EXIT_CODE"
    echo "Check the SLURM output in ${LOGS_DIR}/ for details"
fi

echo "=========================================="
