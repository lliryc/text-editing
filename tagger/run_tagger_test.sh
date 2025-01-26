#!/bin/bash
#SBATCH -p nvidia
#SBATCH --reservation=nlp
# use gpus
#SBATCH --gres=gpu:v100:1
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
pred_mode=train_nopnx


checkpoint=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-500

labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/pnx_sep/nopnx/qalb14+zaebuc_x10/labels.txt
test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/pnx_sep/nopnx/qalb14+zaebuc_x10/train_nopnx.txt


python /home/ba63/arabic-text-editing/tagger/tag.py \
    --file_path $test_file \
    --labels $labels \
    --model_name_or_path $checkpoint \
    --output_dir $checkpoint \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt \
    --rewrite_pred_output_file ${pred_mode}.txt

python tokenize_data.py \
    --input $checkpoint/${pred_mode}.txt  \
    --output $checkpoint/${pred_mode}.txt.tokens \
    --tokenizer_path $checkpoint



