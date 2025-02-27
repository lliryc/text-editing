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


# Train edits pnx segregation
for dataset in qalb14 qalb14+zaebuc_x10 zaebuc; do
    echo "Pnx segregating edits for ${dataset} train..."
    python create_edits_pnx_sep.py \
        --dataset ${dataset}-arabertv02 \
        --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
        --split train \
        --create_edits \
        --create_pnx_edits \
        --edits_granularity subword \
        --input_data_dir ../data/msa-gec/edits/${dataset}/edits_no_compressed \
        --output_data_dir ../data/msa-gec/edits/${dataset}/edits_compressed_pnx_sep

    # Pruning
    for k in 10 20 30; do
        echo "Pruning edits that appear <= ${k} times from ${dataset}..."
        python create_edits_pnx_sep.py \
            --dataset ${dataset}-arabertv02 \
            --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
            --split train \
            --output_data_dir ../data/msa-gec/edits/${dataset}/edits_compressed_pnx_sep \
            --edits_granularity subword \
            --prune \
            --k $k \
            --pruned_output_dir ../data/msa-gec/edits/${dataset}/edits_compressed_pnx_sep_prune_${k}
    done 
done


# Dev edits pnx segregation
for dataset in qalb14 zaebuc; do
    echo "Pnx segregating edits for ${dataset} ${split}..."
    python create_edits_pnx_sep.py \
        --dataset ${dataset}-arabertv02 \
        --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
        --split dev \
        --create_edits \
        --create_pnx_edits \
        --edits_granularity subword \
        --input_data_dir ../data/msa-gec/edits/${dataset}/edits_no_compressed \
        --output_data_dir ../data/msa-gec/edits/${dataset}/edits_compressed_pnx_sep
done