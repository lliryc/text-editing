#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:a100:1
# memory
# SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.%A.out
#SBATCH -e job.%J.%A.err

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=dev
var=qalb14_3


test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_word_level_arabertv02/qalb14/no-compressed/dev.txt


# test_file=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/pnx/${var}/dev_pnx.txt
# test_file=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/nopnx/${var}/dev_nopnx.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-7500/dev.txt.3.tokens
# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/pnx_sep/pnx/zaebuc/dev_pnx.txt
# test_file=/scratch/ba63/arabic-text-editing/coda_data/madar-arabertv02-large/madar/dev.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/zaebuc/dev.txt

sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers_arabertv02/qalb14-word-level-no-compressed
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_word_level_arabertv02/qalb14/no-compressed/labels.txt


# iterative prediction
for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$sys/${pred_mode}.txt.$((i - 1)).tokens
        echo $test_file
    fi

    python tag.py \
        --file_path $test_file \
        --labels $labels \
        --model_name_or_path $sys \
        --output_dir $sys \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

        awk '{for (i=1; i<=NF; i++) print $i; print ""}' $sys/${pred_mode}.txt.${i} > $sys/${pred_mode}.txt.${i}.tokens

done