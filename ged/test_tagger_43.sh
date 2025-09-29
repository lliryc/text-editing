#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:a100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
# SBATCH --mem=50GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

#nvidia-smi
#module purge


####################################
# SEQ LABELING FINE-TUNING SCRIPT
####################################


export DATA_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-ged/modeling-43/pnx_sep/nopnx/dev
# export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# export BERT_MODEL=/scratch/ba63/BERT_models/ARBERTv2
export BERT_MODEL=/Users/kirill.chirkunov/CursorProjects/text-editing/trained_ged_nopnx_model_qalb14_43

export OUTPUT_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-ged/testing-43/pnx_sep/nopnx

export BATCH_SIZE=32
export SEED=42


python3 tag_areta_13.py \
    --tokenized_data_path $DATA_DIR/qalb14-dev-subword-level-areta-43.txt \
    --tokenized_raw_data_path $DATA_DIR/qalb14-dev-subword-level-areta-43.raw.txt \
    --labels $DATA_DIR/qalb14-dev-subword-level-areta-43-labels.txt \
    --model_name_or_path $BERT_MODEL \
    --input_unit subword-level \
    --task msa-ged \
    --output_dir $BERT_MODEL \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file $OUTPUT_DIR/qalb14-dev-subword-level-areta-43-labels.txt \
    --rewrite_pred_output_file $OUTPUT_DIR/qalb14-dev-subword-level-areta-43-rewrite.txt
