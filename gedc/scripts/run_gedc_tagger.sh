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
# GEDC MULTI-HEAD TAGGING SCRIPT
####################################

# Set environment variables
export PYTHONPATH=/Users/kirill.chirkunov/CursorProjects/text-editing
export DATA_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/data/msa-gedc/modeling-mh/pnx_sep/nopnx/train
export OUTPUT_DIR=/Users/kirill.chirkunov/CursorProjects/text-editing/output_data
export BERT_MODEL=aubmindlab/bert-base-arabertv02

# Training parameters
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export SEED=42

# Run the GEDC multi-head tagger
python3 gedc/tag_multi_label_areta.py \
    --tokenized_data_path $DATA_DIR/qalb14-train-subword-level-multi-heads.txt \
    --optim adamw_torch \
    --edit_labels $DATA_DIR/edit-labels.txt \
    --areta13_labels $DATA_DIR/areta-13-labels.txt \
    --areta43_labels $DATA_DIR/areta-43-labels.txt \
    --label_pred_output_file $OUTPUT_DIR/qalb14-train-subword-level-multi-heads-labels.txt \
    --rewrite_pred_output_file $OUTPUT_DIR/qalb14-train-subword-level-multi-heads-rewrite.txt \
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
