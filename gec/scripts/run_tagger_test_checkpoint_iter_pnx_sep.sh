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
# pred_mode=test_qalb15_L1
pred_mode=test


checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-7000
checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-pnx-prune-10-a100/checkpoint-500

labels_nopnx=/home/ba63/text-editing/data/msa-gec/modeling/qalb14+zaebuc_x10/pnx_sep/qalb14+zaebuc_x10-nopnx-prune-30/compressed/subword-level/labels.txt
labels_pnx=/home/ba63/text-editing/data/msa-gec/modeling/qalb14+zaebuc_x10/pnx_sep/qalb14+zaebuc_x10-pnx-prune-10/compressed/subword-level/labels.txt

test_file=/home/ba63/text-editing/data/msa-gec/modeling/zaebuc/zaebuc/test.txt
test_file_raw=/home/ba63/text-editing/data/msa-gec/modeling/zaebuc/zaebuc/test.raw.txt



for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint_nopnx/${pred_mode}.txt.$((i - 1)).check.tokens
        test_file_raw=$checkpoint_nopnx/${pred_mode}.txt.$((i - 1)).check.raw.tokens
        echo $test_file
    fi

    # Running GEC without pnx correction
    python tag.py \
        --tokenized_data_path $test_file \
        --tokenized_raw_data_path $test_file_raw \
        --labels $labels_nopnx \
        --input_unit subword-level \
        --task msa-gec \
        --model_name_or_path $checkpoint_nopnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i}.check \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}.check

    # Tokenizing the output
    python utils/tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.${i}.check  \
        --tokenizer_path $checkpoint_nopnx
done


# Running Pnx GEC model on the output
python tag.py \
    --tokenized_data_path $checkpoint_nopnx/${pred_mode}.txt.2.check.tokens \
    --tokenized_raw_data_path $checkpoint_nopnx/${pred_mode}.txt.2.check.raw.tokens \
    --labels $labels_pnx \
    --input_unit subword-level \
    --task msa-gec \
    --model_name_or_path $checkpoint_pnx \
    --output_dir $checkpoint_nopnx \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt.2.pnx_edit.check \
    --rewrite_pred_output_file ${pred_mode}.txt.2.pnx_edit.check