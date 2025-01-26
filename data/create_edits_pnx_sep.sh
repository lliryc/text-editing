#!/bin/bash
# Set number of tasks to run
#SBATCH -q nlp
#SBATCH -n 2
#SBATCH -c 50
# SBATCH --mem=50GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err



# python create_edits_pnx_sep.py \
#     --dataset qalb14+zaebuc_x10 \
#     --split train \
#     --create_edits \
#     --input_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_no_compressed_pred \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_pnx_sep_pred


python create_edits_pnx_sep.py \
    --dataset  qalb14+zaebuc_x10 \
    --split train \
    --prune \
    --k 30 \
    --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_pnx_sep_pred \
    --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_pnx_sep_pred_prune_30


# python create_edits_pnx_sep.py \
#     --dataset qalb14 \
#     --split train \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/mle/edits_compressed_pnx_sep \
#     --prune \
#     --k 30 \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/mle/edits_compressed_pnx_sep_prune_30





# for k in 10 20 30
#     do
#         python create_edits_pnx_sep.py \
#             --split train \
#             --output_data_dir /scratch/ba63/arabic-text-editing/edits/morph/edits_compressed_pnx_sep \
#             --token 3 \
#             --prune \
#             --prune_cor \
#             --k $k \
#             --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/morph/edits_compressed_pnx_sep_prune_${k}_cor
#     done
