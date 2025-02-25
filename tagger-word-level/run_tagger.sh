#!/bin/bash
#SBATCH -p nvidia
# SBATCH -q nlp
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

export DATA_DIR=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_word_level_arabertv02/compressed
# export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
# export BERT_MODEL=/scratch/ba63/BERT_models/ARBERTv2
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabertv02
export OUTPUT_DIR=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers_arabertv02_15_iter/qalb14+zaebuc_x10-word-level-a100

export BATCH_SIZE=32
# export NUM_EPOCHS=10
export NUM_EPOCHS=15
export SAVE_STEPS=500
export SEED=42


python tag.py \
    --file_path $DATA_DIR/train.txt \
    --optim adamw_torch \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --report_to "none" \
    --overwrite_output_dir