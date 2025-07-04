#!/bin/bash 

#SBATCH --job-name=teirv_inference_only
#SBATCH --output=outputs/logfiles/teirv_inference_%j.out
#SBATCH --error=outputs/logfiles/teirv_inference_%j.err
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00 
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# TEIRV NPE Inference Only SLURM Script
# 
# This script runs only the inference step using an existing trained model:
# - Uses existing model from production_run_20250703_123040
# - Skips data generation and training
# - Runs inference on clinical patient data with device compatibility fix
#
# Usage: sbatch scripts/slurm_scripts/run_inference_only.sh

echo "=========================================="
echo "TEIRV NPE Inference Only"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Using existing model: workflows/production_run_20250703_123040/npe_model.pkl"
echo "=========================================="

# Create output directories
mkdir -p outputs/logfiles

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

# Run inference only
echo "Starting inference at $(date)"

time python -u run_inference_only.py

INFERENCE_EXIT_CODE=$?

echo "=========================================="
echo "Inference completed at $(date)"
echo "Exit code: $INFERENCE_EXIT_CODE"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "✅ INFERENCE COMPLETED SUCCESSFULLY"
    echo "📁 Results available in: workflows/production_run_20250703_123040/inference_results/"
    echo ""
    echo "Generated files:"
    echo "  - Patient-specific corner plots: patient_*_corner.png"
    echo "  - Patient-specific predictive plots: patient_*_predictive.png"
    echo "  - Parameter estimates: clinical_parameter_estimates.csv"
    echo ""
    echo "To view results:"
    echo "  ls -la workflows/production_run_20250703_123040/inference_results/"
    echo "  cat workflows/production_run_20250703_123040/inference_results/clinical_parameter_estimates.csv"
else
    echo "❌ INFERENCE FAILED with exit code $INFERENCE_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - Standard output: outputs/logfiles/teirv_inference_${SLURM_JOB_ID}.out"
    echo "  - Standard error: outputs/logfiles/teirv_inference_${SLURM_JOB_ID}.err"
fi

echo "=========================================="