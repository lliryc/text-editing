#!/bin/bash
# Set number of tasks to run
#SBATCH -p compute
#SBATCH -n 2
#SBATCH -c 50
# SBATCH --mem=50GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


python edits_ensemble.py \
    --input_data /home/ba63/gec-release/data/gec/modeling/qalb14/wo_camelira/full/dev.json \
    --models_outputs /scratch/ba63/arabic-text-editing/ensemble/dev.txt.2 /scratch/ba63/arabic-text-editing/ensemble/dev.txt.nopnx_edit.2.pnx_edit.pred_t05 /scratch/ba63/arabic-text-editing/ensemble/qalb14_dev.preds.txt 