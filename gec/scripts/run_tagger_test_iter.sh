#!/usr/bin/env bash
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
pred_mode=test

# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/dev.txt
# test_file_raw=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/dev.raw.txt

# checkpoint=/scratch/ba63/arabic-text-editing/gec_taggers/zaebuc_x10/taggers_arabertv02/zaebuc_x10-a100
# labels=/scratch/ba63/arabic-text-editing/gec_data/zaebuc_x10/tagger_data_arabertv02/zaebuc_x10/labels.txt

test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/test.txt
test_file_raw=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/test.raw.txt

checkpoint=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers_arabertv02_15_iter/qalb14+zaebuc_x10-prune-30-a100/checkpoint-8500
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_arabertv02/qalb14+zaebuc_x10-prune-30/compressed/labels.txt

for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint/${pred_mode}.txt.$((i - 1)).tokens
        test_file_raw=$checkpoint/${pred_mode}.txt.$((i - 1)).raw.tokens
        echo $test_file
    fi

    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --tokenized_data_path $test_file \
        --tokenized_raw_data_path $test_file_raw \
        --labels $labels \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

        python tokenize_data.py \
            --input $checkpoint/${pred_mode}.txt.${i}  \
            --tokenizer_path $checkpoint
done


