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

python run_gpt.py \
    --output_dir /scratch/ba63/arabic-text-editing/llms-outputs/gpt4o-outputs/few-shot-en \
    --n_shot 5 \
    --model gpt-4o \
    --lang en \
    --split test \
    --task gec
