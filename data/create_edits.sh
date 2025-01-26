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


# python create_edits.py \
#     --split train \
#     --dataset qalb14+zaebuc_x10 \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed \
#     --k 30 \
#     --prune \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed_prune_30


# python create_edits.py \
#     --split train \
#     --dataset qalb14 \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14/qalb14_train.src.txt \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14/qalb14_train.tgt.txt \
#     --create_edits \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed

# python create_edits.py \
#     --split train \
#     --dataset zaebuc \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/gec/zaebuc/zaebuc_train.src.txt \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/gec/zaebuc/zaebuc_train.tgt.txt \
#     --create_edits \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed


python create_edits.py \
    --dataset qalb14-arabertv02 \
    --split dev \
    --src_file_path /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
    --tgt_file_path /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
    --create_edits \
    --edits_granularity word \
    --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_no_compressed \
    --compress \
    --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed


# python create_edits.py \
#     --split dev \
#     --src_file_path /scratch/ba63/arabic-text-editing/gec_data/qalb14/mle_processed/qalb14_dev.src.mle.txt \
#     --tgt_file_path /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
#     --create_edits \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/mle/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/mle/edits_compressed
    

# pruning
# k=30
# python create_edits.py \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/mix/edits_compressed \
#     --split train \
#     --k $k \
#     --prune \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/mix/edits_compressed_prune_${k}


# qalb14+qalb15+zaebuc
# python create_edits.py \
#     --split train \
#     --dataset qalb14+15+zaebuc \
#     --create_edits \
#     --input_data_dir /home/ba63/gec-release/data/gec/modeling/mix/wo_camelira/full  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed


# python create_edits.py \
#     --split train \
#     --dataset qalb14+15+zaebuc \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed \
#     --prune \
#     --k 30 \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+15+zaebuc/edits_compressed_prune_30
    




# qalb14+Zaebuc
# python create_edits.py \
#     --split train \
#     --dataset qalb14+zaebuc \
#     --create_edits \
#     --input_data_dir /scratch/ba63/arabic-text-editing/edits/qalb14+zaebuc/gec  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/qalb14+zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/qalb14+zaebuc/edits_compressed




# qalb14+15
# python create_edits.py \
#     --split train \
#     --create_edits \
#     --input_data_dir /home/ba63/gec-release/data/gec/modeling/qalb14-15/wo_camelira/full  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/qalb14+15/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/qalb14+15/edits_compressed


# qalb14+15+zaebuc
# python create_edits.py \
#     --split train \
#     --create_edits \
#     --input_data_dir /home/ba63/gec-release/data/gec/modeling/mix/wo_camelira/full  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/qalb14+15+zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/qalb14+15+zaebuc/edits_compressed


# zaebuc
# python create_edits.py \
#     --dataset zaebuc \
#     --split dev \
#     --create_edits \
#     --input_data_dir /home/ba63/gec-release/data/gec/modeling/zaebuc/wo_camelira/full  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/zaebuc/edits_compressed


# qalb15
# Make sure to preprocess the qalb15 data before creating edits. Some words like ؼتر
# will generate unk subwords! These should be normalized using the arclean charmapper in cameltools

# python create_edits.py \
#     --dataset qalb15 \
#     --split dev \
#     --create_edits \
#     --input_data_dir /home/ba63/gec-release/data/gec/modeling/qalb15/wo_camelira/full  \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/qalb15/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/qalb15/edits_compressed
