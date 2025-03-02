#!/bin/bash
# Set number of tasks to run
# SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

python run_fanar.py \
    --output_dir /scratch/ba63/arabic-text-editing/llms-outputs/fanar-outputs-new/few-shot-en \
    --n_shot 5 \
    --lang en \
    --model Fanar-C-1-8.7B \
    --task coda