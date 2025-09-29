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

nvidia-smi
module purge


####################################
# SEQ LABELING FINE-TUNING SCRIPT
####################################


export DATA_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-ged/modeling-13/pnx_sep/nopnx/train
# export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# export BERT_MODEL=/scratch/ba63/BERT_models/ARBERTv2
export BERT_MODEL=aubmindlab/bert-base-arabertv02

export OUTPUT_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/output_data

export BATCH_SIZE=32
export NUM_EPOCHS=10 # 15 for qalb14+zaebuc_x10
export SAVE_STEPS=500
export SEED=42


python3 tag_areta_13.py \
    --tokenized_data_path $DATA_DIR/qalb14-train-subword-level-areta-13.txt \
    --optim adamw_torch \
    --labels $DATA_DIR/qalb14-train-subword-level-areta-13-labels.txt \
    --model_name_or_path $BERT_MODEL \
    --input_unit subword-level \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --report_to "none" \
    --overwrite_output_dir