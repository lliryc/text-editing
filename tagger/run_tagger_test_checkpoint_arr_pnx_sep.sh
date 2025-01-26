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
#SBATCH --array=0-11

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=dev
var=qalb14

test_file=/scratch/ba63/arabic-text-editing/tagger_data/$var/dev.txt

sys_nopnx=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_10/$var
sys_pnx=/scratch/ba63/arabic-text-editing/pnx_taggers/pnx_prune_10/$var

labels_nopnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_10/nopnx/$var/labels.txt
labels_pnx=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep_prune_10/pnx/$var/labels.txt


# Checkpoints
create_checkpoints() {
  local sys_dir=$1
  echo "${sys_dir}" \
       "${sys_dir}/checkpoint-500" \
       "${sys_dir}/checkpoint-1000" \
       "${sys_dir}/checkpoint-1500" \
       "${sys_dir}/checkpoint-2000" \
       "${sys_dir}/checkpoint-2500" \
       "${sys_dir}/checkpoint-3000" \
       "${sys_dir}/checkpoint-3500" \
       "${sys_dir}/checkpoint-4000" \
       "${sys_dir}/checkpoint-4500" \
       "${sys_dir}/checkpoint-5000" \
       "${sys_dir}/checkpoint-5500" \
       "${sys_dir}/checkpoint-6000"
}

# Copy tokenizer files
copy_tokenizer_files() {
  local source_dir=$1
  local target_dir=$2
  cp ${source_dir}/tokenizer_config.json ${target_dir}
  cp ${source_dir}/vocab.txt ${target_dir}
  cp ${source_dir}/special_tokens_map.json ${target_dir}
}

checkpoints_nopnx=($(create_checkpoints $sys_nopnx))
checkpoints_pnx=($(create_checkpoints $sys_pnx))

checkpoint_nopnx=${checkpoints_nopnx[$SLURM_ARRAY_TASK_ID]}
checkpoint_pnx=${checkpoints_pnx[$SLURM_ARRAY_TASK_ID]}

copy_tokenizer_files $sys_nopnx $checkpoint_nopnx
copy_tokenizer_files $sys_pnx $checkpoint_pnx


printf "Running evaluation using ${checkpoint_nopnx} and ${checkpoint_pnx}..\n"


for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.$((i - 1)).pnx_edit.tok.check
    fi

    # Running GEC without pnx correction
    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --file_path $test_file \
        --labels $labels_nopnx \
        --model_name_or_path $checkpoint_nopnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.nopnx_edit.${i}.check \
        --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.check

    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.check  \
        --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.check \
        --tokenizer_path $checkpoint_nopnx


    # Running Pnx GEC model on the output
    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --file_path $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.tok.check \
        --labels $labels_pnx \
        --model_name_or_path $checkpoint_pnx \
        --output_dir $checkpoint_nopnx \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.nopnx_edit.pnx_edit.${i}.check \
        --rewrite_pred_output_file ${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.check


    # Tokenizing the output
    python tokenize_data.py \
        --input $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.check  \
        --output $checkpoint_nopnx/${pred_mode}.txt.nopnx_edit.${i}.pnx_edit.tok.check \
        --tokenizer_path $checkpoint_nopnx


done