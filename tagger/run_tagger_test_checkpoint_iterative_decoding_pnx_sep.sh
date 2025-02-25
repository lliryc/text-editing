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


# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-pnx-prune-20-a100/checkpoint-500

# labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/pnx_sep/qalb14-nopnx-prune-30/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/pnx_sep/qalb14-pnx-prune-20/labels.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/qalb14/compressed/test_qalb15_L1.txt
# test_file_raw=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/qalb14/compressed/test_qalb15_L1.raw.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/qalb14/compressed/test.txt
# test_file_raw=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_arabertv02/qalb14/compressed/test.raw.txt


# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-4500
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02/qalb14+zaebuc_x10-pnx-prune-10-a100/checkpoint-500
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14_adj+zaebuc_x10/pnx_taggers_arabertv02/qalb14_adj+zaebuc_x10-pnx-prune-10-a100/checkpoint-500

# labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_arabertv02/pnx_sep/qalb14+zaebuc_x10-nopnx-prune-30/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_arabertv02/pnx_sep/qalb14+zaebuc_x10-pnx-prune-10/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14_adj+zaebuc_x10/tagger_data_arabertv02/pnx_sep/qalb14_adj+zaebuc_x10-pnx-prune-10/labels.txt

checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-7000
checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-pnx-prune-10-a100/checkpoint-500

labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_arabertv02/pnx_sep/qalb14+zaebuc_x10-nopnx-prune-30/labels.txt
labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_arabertv02/pnx_sep/qalb14+zaebuc_x10-pnx-prune-10/labels.txt

test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/test.txt
test_file_raw=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_arabertv02/zaebuc/test.raw.txt



for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint_nopnx/${pred_mode}.txt.$((i - 1)).tokens
        test_file_raw=$checkpoint_nopnx/${pred_mode}.txt.$((i - 1)).raw.tokens
        echo $test_file
    fi

    # Running GEC without pnx correction
    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --tokenized_data_path $test_file \
        --tokenized_raw_data_path $test_file_raw \
        --labels $labels_nopnx \
        --model_name_or_path $checkpoint_nopnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.${i}  \
        --tokenizer_path $checkpoint_nopnx
done


# Running Pnx GEC model on the output
python /home/ba63/arabic-text-editing/tagger/tag.py \
    --tokenized_data_path $checkpoint_nopnx/${pred_mode}.txt.2.tokens \
    --tokenized_raw_data_path $checkpoint_nopnx/${pred_mode}.txt.2.raw.tokens \
    --labels $labels_pnx \
    --model_name_or_path $checkpoint_pnx \
    --output_dir $checkpoint_nopnx \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt.2.pnx_edit \
    --rewrite_pred_output_file ${pred_mode}.txt.2.pnx_edit