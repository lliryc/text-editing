#!/usr/bin/env bash
#SBATCH -p nvidia
#SBATCH --reservation=nlp
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
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



checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/nopnx_prune_20_mt/qalb14_a0.2/checkpoint-1500
checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20_mt/qalb14_a0.2/checkpoint-500

labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/nopnx/qalb14/labels.txt
labels_nopnx_bin=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/nopnx/qalb14/labels.bin.txt

labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/pnx/qalb14/labels.txt
labels_pnx_bin=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/pnx/qalb14/labels.bin.txt



test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/qalb14_mt/dev.bin.txt



for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.$((i - 1)).pnx_edit.tok.pred_t05
    fi

    

    # Running GEC without pnx correction
    python tag_new.py \
        --file_path $test_file \
        --labels $labels_nopnx \
        --binary_labels $labels_nopnx_bin \
        --model_name_or_path $checkpoint_nopnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.nopnx_edit.${i}.pred_t05 \
        --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.pred_t05

    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pred_t05 \
        --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.pred_t05 \
        --tokenizer_path $checkpoint_nopnx


    # Running Pnx GEC model on the output
    python tag_new.py \
        --file_path $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.pred_t05 \
        --labels $labels_pnx \
        --binary_labels $labels_pnx_bin \
        --model_name_or_path $checkpoint_pnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --pred_threshold 0.5 \
        --label_pred_output_file ${pred_mode}.preds.txt.nopnx_edit.pnx_edit.${i}.pred_t05 \
        --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.pred_t05


    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.pred_t05  \
        --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.tok.pred_t05 \
        --tokenizer_path $checkpoint_nopnx


done