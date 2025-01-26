#!/bin/bash
#SBATCH -p nvidia
#SBATCH --reservation=nlp
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
# SBATCH --mem=50GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


nvidia-smi
module purge


####################################
# ERROR DETECTION FINE-TUNING SCRIPT
####################################

export DATA_DIR=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/nopnx/qalb14
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
export OUTPUT_DIR=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/nopnx_prune_20_mt/qalb14_a0.2_lbl_smthng


export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export SEED=42


python tag_new.py \
    --file_path $DATA_DIR/train_nopnx.bin.txt \
    --optim adamw_torch \
    --labels $DATA_DIR/labels.txt \
    --binary_labels $DATA_DIR/labels.bin.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --overwrite_output_dir
