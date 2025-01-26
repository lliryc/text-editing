#!/bin/bash
#SBATCH -p nvidia
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
pred_mode=dev


# sys=/scratch/ba63/arabic-text-editing/pnx_taggers/pnx_prune_20/qalb14/checkpoint-1000
# DATA_DIR=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/pnx/qalb14

# sys=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_20/qalb14/checkpoint-1500
# DATA_DIR=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/nopnx/qalb14
# LABELS=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_20/nopnx/qalb14/labels.txt

sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+15+zaebuc/pnx_taggers/nopnx_prune_10/qalb14+15+zaebuc
DATA_DIR=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/pnx_sep/nopnx/zaebuc
LABELS=/scratch/ba63/arabic-text-editing/gec_data/qalb14+15+zaebuc/pnx_sep_prune_10/nopnx/qalb14+15+zaebuc/labels.txt

# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2


printf "Running evaluation using ${sys}..\n"

python /home/ba63/arabic-text-editing/tagger/tag.py \
    --file_path $DATA_DIR/dev_nopnx.txt \
    --labels $LABELS \
    --model_name_or_path $sys \
    --output_dir $sys \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --debug_mode \
    --topk_pred \
    --do_pred \
    --rewrite_pred_output_file ${pred_mode}.txt.debug


