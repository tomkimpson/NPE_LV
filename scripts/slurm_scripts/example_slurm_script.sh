#!/bin/bash 
 
#SBATCH --job-name=teirv_jax_test3
#SBATCH --output=outputs/logfiles/teirv_jax_test3_output.txt
#SBATCH --export=ALL 
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00 
#SBATCH --mem=4G


source ~/.bashrc
conda activate NPE_LV_JAX
time python -u src/TEIRV/teirv_simulator_jax.py

