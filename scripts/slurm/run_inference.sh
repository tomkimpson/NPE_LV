#!/bin/bash

#SBATCH --job-name=teirv_inference_%j
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# TEIRV NPE Inference-Only SLURM Script
#
# This script runs inference on clinical patient data using a pre-trained NPE model.
# Useful for re-running predictions without retraining.
#
# Usage: sbatch scripts/slurm/run_inference.sh
#
# Configuration:
#   - MODEL_PATH: Path to trained NPE model
#   - CLINICAL_DATA_DIR: Directory containing patient SSV files
#   - N_SAMPLES: Number of posterior samples per patient
#   - OUTPUT_DIR: Where to save inference results

# Configuration - EDIT THESE PATHS
MODEL_PATH="results/run_20251017_160237/model/npe_model.pkl"
CLINICAL_DATA_DIR="data/clinical"
N_SAMPLES=20000
OUTPUT_DIR="results/inference_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "TEIRV NPE Inference-Only"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Model: $MODEL_PATH"
echo "Clinical data: $CLINICAL_DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Posterior samples: $N_SAMPLES"
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

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Verify clinical data directory exists
if [ ! -d "$CLINICAL_DATA_DIR" ]; then
    echo "❌ ERROR: Clinical data directory not found at $CLINICAL_DATA_DIR"
    exit 1
fi

# Run inference
echo "Starting inference at $(date)"
echo "Command: python -u src/main.py --device cuda inference --model $MODEL_PATH --data_dir $CLINICAL_DATA_DIR --output $OUTPUT_DIR --n_samples $N_SAMPLES"
echo ""

time python -u src/main.py --device cuda inference \
    --model "$MODEL_PATH" \
    --data_dir "$CLINICAL_DATA_DIR" \
    --output "$OUTPUT_DIR" \
    --n_samples $N_SAMPLES \
    --min_detections 5 \
    --min_peak_vl 2.0

INFERENCE_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Inference completed at $(date)"
echo "Exit code: $INFERENCE_EXIT_CODE"

# Move SLURM output file to results directory
SLURM_OUTPUT="slurm-${SLURM_JOB_ID}.out"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "✅ INFERENCE COMPLETED SUCCESSFULLY"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"

    # Move SLURM log to output directory
    if [ -f "$SLURM_OUTPUT" ]; then
        mv "$SLURM_OUTPUT" "$OUTPUT_DIR/" 2>/dev/null
        echo "SLURM log moved to: ${OUTPUT_DIR}/${SLURM_OUTPUT}"
    fi
else
    echo "❌ INFERENCE FAILED with exit code $INFERENCE_EXIT_CODE"
    echo "Check the SLURM output for details"
fi

echo "=========================================="
