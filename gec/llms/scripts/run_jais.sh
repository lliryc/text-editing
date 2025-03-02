#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:a100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


python run_jais.py \
    --output_dir /scratch/ba63/arabic-text-editing/llms-outputs/jais-outputs/few-shot-ar \
    --n_shot 5 \
    --lang ar \
    --task gec