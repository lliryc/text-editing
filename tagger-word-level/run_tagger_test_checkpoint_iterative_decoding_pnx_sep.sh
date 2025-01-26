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


# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_20/qalb14/checkpoint-1500
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/pnx_taggers/pnx_prune_20/qalb14/checkpoint-1000

# labels_nopnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_20/nopnx/qalb14/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_20/pnx/qalb14/labels.txt

# =====================================================================================

# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_30_s1234/qalb14/checkpoint-3000
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/pnx_taggers/pnx_prune_20_s1234/qalb14_5/checkpoint-1000
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/pnx_taggers/pnx_s1234/qalb14_3/checkpoint-1000

# labels_nopnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_30/nopnx/qalb14/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_20/pnx/qalb14_5/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/pnx/qalb14_3/labels.txt

# =====================================================================================


# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+15+zaebuc/pnx_taggers/nopnx_prune_10/qalb14+15+zaebuc
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14_adj+zaebuc_x10/pnx_taggers/pnx_prune_20/checkpoint-3500

# labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+15+zaebuc/pnx_sep_prune_10/nopnx/qalb14+15+zaebuc/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14_adj+zaebuc_x10/pnx_sep_prune_20/pnx/qalb14_adj+zaebuc_x10/labels.txt

# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14_adj+zaebuc_x10/pnx_taggers/nopnx_prune_20/checkpoint-6000
# checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers/pnx_prune_20/checkpoint-2500

# labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14_adj+zaebuc_x10/pnx_sep_prune_20/nopnx/qalb14_adj+zaebuc_x10/labels.txt
# labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/pnx_sep_prune_20/pnx/qalb14+zaebuc_x10/labels.txt


# test_file=/scratch/ba63/arabic-text-editing/tagger_data/qalb14/dev.txt
# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/zaebuc/dev.txt
# test_file=/home/ba63/arabic-text-editing/mle/zaebuc_dev.mle.txt.tok


# for i in {1..3}
# do
#     if [[ $i != 1 ]]; then
#         test_file=$checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.$((i - 1)).pnx_edit.tok.pred_t05.check
#     fi

#     # Running GEC without pnx correction
#     python /home/ba63/arabic-text-editing/tagger/tag.py \
#         --file_path $test_file \
#         --labels $labels_nopnx \
#         --model_name_or_path $checkpoint_nopnx \
#         --output_dir $checkpoint_nopnx \
#         --per_device_eval_batch_size $BATCH_SIZE \
#         --seed $SEED \
#         --do_pred \
#         --label_pred_output_file ${pred_mode}.preds.txt.${i}.pred_t05.check \
#         --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.pred_t05.check

#     # Tokenizing the output
#     python tokenize_data.py \
#         --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pred_t05.check  \
#         --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.pred_t05.check \
#         --tokenizer_path $checkpoint_nopnx


#     # Running Pnx GEC model on the output
#     python /home/ba63/arabic-text-editing/tagger/tag.py \
#         --file_path $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.pred_t05.check \
#         --labels $labels_pnx \
#         --model_name_or_path $checkpoint_pnx \
#         --output_dir $checkpoint_nopnx \
#         --per_device_eval_batch_size $BATCH_SIZE \
#         --seed $SEED \
#         --do_pred \
#         --pred_threshold 0.5 \
#         --label_pred_output_file ${pred_mode}.preds.txt.nopnx_edit.pnx_edit.${i}.pred_t05.check \
#         --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.pred_t05.check


#     # Tokenizing the output
#     python tokenize_data.py \
#         --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.pred_t05.check \
#         --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.tok.pred_t05.check \
#         --tokenizer_path $checkpoint_nopnx
# done



checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-7500
labels_nopnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/pnx_sep/nopnx/qalb14+zaebuc_x10/labels.txt

checkpoint_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/pnx_prune_10/checkpoint-6000
labels_pnx=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/pnx_sep_prune_10/pnx/qalb14+zaebuc_x10/labels.txt


fname=dev.txt.3.tokens
test_file=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-7500/$fname


for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint_nopnx/${fname}.txt.pnx_edit.$((i - 1)).tok.pred_t05
    fi

    # Running Pnx GEC model on the output
    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --file_path $test_file \
        --labels $labels_pnx \
        --model_name_or_path $checkpoint_pnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --pred_threshold 0.5 \
        --label_pred_output_file ${fname}.preds.pnx_edit.${i}.pred_t05 \
        --rewrite_pred_output_file ${fname}.txt.pnx_edit.${i}.pred_t05


    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${fname}.txt.pnx_edit.${i}.pred_t05 \
        --output $checkpoint_nopnx/${fname}.txt.pnx_edit.${i}.tok.pred_t05 \
        --tokenizer_path $checkpoint_nopnx
done