#!/bin/bash
#SBATCH --job-name=teirv_resume
#SBATCH --output=outputs/logfiles/teirv_resume_%j.out
#SBATCH --error=outputs/logfiles/teirv_resume_%j.err
#SBATCH --export=ALL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Resume TEIRV training using existing 100k data
# Skip expensive data generation, complete Phase 2 pipeline
# 
# This script uses the data generated from the previous run:
# workflows/production_run_20250704_134546/training_data.pkl
#
# Usage: sbatch scripts/slurm_scripts/resume_teirv_training.sh

echo "==========================================
TEIRV NPE Resume Training (Phase 2)
=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Using existing data: workflows/production_run_20250704_134546/training_data.pkl"
echo "Phase 2 config: 500 epochs, 100k samples, 50 epoch patience"
echo "=========================================="

# Create output directories
mkdir -p outputs/logfiles
mkdir -p workflows

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

# Check if training data exists
if [ ! -f "workflows/production_run_20250704_134546/training_data.pkl" ]; then
    echo "❌ ERROR: Training data not found!"
    echo "Expected: workflows/production_run_20250704_134546/training_data.pkl"
    exit 1
fi

echo "✅ Training data found: $(stat -c%s workflows/production_run_20250704_134546/training_data.pkl | numfmt --to=iec) bytes"

# Train model with existing 100k data
echo ""
echo "🧠 Step 1: Training NPE model with Phase 2 optimizations..."
echo "=========================================="
time python -u scripts/TEIRV_workflow.py train \
    --data workflows/production_run_20250704_134546/training_data.pkl \
    --output workflows/production_run_20250704_134546/npe_model_phase2.pkl \
    --max_epochs 500 \
    --early_stopping 50 \
    --learning_rate 5e-4 \
    --batch_size 512 \
    --hidden_features 256 \
    --num_transforms 8

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully"
    echo ""
    
    # Run clinical inference
    echo "🏥 Step 2: Running clinical inference..."
    echo "=========================================="
    time python -u scripts/TEIRV_workflow.py inference \
        --model workflows/production_run_20250704_134546/npe_model_phase2.pkl \
        --output workflows/production_run_20250704_134546/inference_results \
        --n_samples 20000 \
        --min_detections 5 \
        --min_peak_vl 2.0
    
    INFERENCE_EXIT_CODE=$?
    
    echo ""
    echo "=========================================="
    echo "Pipeline completed at $(date)"
    echo "Training exit code: $TRAIN_EXIT_CODE"
    echo "Inference exit code: $INFERENCE_EXIT_CODE"
    echo "Results available in: workflows/production_run_20250704_134546/"
    echo "=========================================="
    
    if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
        echo "🎉 Phase 2 optimization pipeline completed successfully!"
        echo ""
        echo "Generated files:"
        echo "  - npe_model_phase2.pkl (trained model)"
        echo "  - inference_results/ (clinical results)"
        echo ""
        echo "To view results:"
        echo "  ls -la workflows/production_run_20250704_134546/"
        echo "  ls -la workflows/production_run_20250704_134546/inference_results/"
    else
        echo "❌ Inference failed with exit code $INFERENCE_EXIT_CODE"
    fi
    
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs for details"
fi

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="