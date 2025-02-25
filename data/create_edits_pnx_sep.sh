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
#     --dataset qalb14-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --create_edits \
#     --create_pnx_edits \
#     --edits_granularity subword \
#     --input_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_no_compressed \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed_pnx_sep

for k in 10 20 30
do
    python create_edits_pnx_sep.py \
        --dataset qalb14-arabertv02 \
        --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
        --split train \
        --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed_pnx_sep \
        --edits_granularity subword \
        --prune \
        --k $k \
        --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed_pnx_sep_prune_${k}
done

# python create_edits_pnx_sep.py \
#     --dataset zaebuc-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --create_edits \
#     --create_pnx_edits \
#     --edits_granularity subword \
#     --input_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_no_compressed \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed_pnx_sep

# for k in 10 20 30
# do
#     python create_edits_pnx_sep.py \
#         --dataset zaebuc-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed_pnx_sep \
#         --edits_granularity subword \
#         --prune \
#         --k $k \
#         --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed_pnx_sep_prune_${k}
# done



# python create_edits_pnx_sep.py \
#     --dataset qalb14+15+zaebuc-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --create_edits \
#     --create_pnx_edits \
#     --edits_granularity subword \
#     --input_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_no_compressed \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed_pnx_sep

# for k in 10 20 30
# do
#     python create_edits_pnx_sep.py \
#         --dataset qalb14+15+zaebuc-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed_pnx_sep \
#         --edits_granularity subword \
#         --prune \
#         --k $k \
#         --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed_pnx_sep_prune_${k}
# done


# python create_edits_pnx_sep.py \
#     --dataset qalb14+zaebuc_x10-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split train \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_pnx_sep \
#     --edits_granularity subword \
#     --prune \
#     --k 10 \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_pnx_sep_prune_10