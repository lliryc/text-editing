#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:a100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=test


# checkpoint=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500

# test_file=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data_arabertv02/madar/compressed/test.txt
# test_file_raw=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data_arabertv02/madar/compressed/test.raw.txt
# labels=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data_arabertv02/madar-prune-10/compressed/labels.txt


checkpoint=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers/madar-prune-10-a100/checkpoint-1500


test_file=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data/madar/compressed/test.txt
test_file_raw=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data/madar/compressed/test.raw.txt
labels=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data/madar-prune-10/compressed/labels.txt


python /home/ba63/arabic-text-editing/tagger/tag.py \
    --tokenized_data_path $test_file \
    --tokenized_raw_data_path $test_file_raw \
    --labels $labels \
    --model_name_or_path $checkpoint \
    --output_dir $checkpoint \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt \
    --rewrite_pred_output_file ${pred_mode}.txt


