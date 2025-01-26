#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH --reservation=nlp
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
#SBATCH --array=0-12

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=dev


test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/qalb14_mt/dev.bin.txt
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/nopnx/qalb14/labels.txt
labels_bin=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20_mt/nopnx/qalb14/labels.bin.txt

sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/nopnx_prune_20_mt/qalb14_a0.2_lbl_smthng

# Array of checkpoints
checkpoints=(
  "${sys}"
  "${sys}/checkpoint-500"
  "${sys}/checkpoint-1000"
  "${sys}/checkpoint-1500"
  "${sys}/checkpoint-2000"
  "${sys}/checkpoint-2500"
  "${sys}/checkpoint-3000"
  "${sys}/checkpoint-3500"
  "${sys}/checkpoint-4000"
  "${sys}/checkpoint-4500"
  "${sys}/checkpoint-5000"
  "${sys}/checkpoint-5500"
  "${sys}/checkpoint-6000"
)


checkpoint=${checkpoints[$SLURM_ARRAY_TASK_ID]}

cp $sys/tokenizer_config.json $checkpoint
cp $sys/vocab.txt $checkpoint
cp $sys/special_tokens_map.json $checkpoint

printf "Running evaluation using ${checkpoint}..\n"

# iterative prediction
for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint/${pred_mode}.txt.$((i - 1)).tokens
        echo $test_file
    fi

    python tag_new.py \
        --file_path $test_file \
        --labels $labels \
        --binary_labels $labels_bin \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

        python tokenize_data.py \
            --input $checkpoint/${pred_mode}.txt.${i}  \
            --output $checkpoint/${pred_mode}.txt.${i}.tokens \
            --tokenizer_path $checkpoint
done




# # Evaluation
# paste $DATA_DIR/${pred_mode}.txt $checkpoint/${pred_mode}.preds.txt${topk_pred} \
#     > $checkpoint/eval_data_${pred_mode}.txt

# python /home/ba63/arabic-text-editing/tagger/evaluate.py \
#     --data $checkpoint/eval_data_${pred_mode}.txt \
#     --labels $LABELS \
#     --output $checkpoint/${pred_mode}.results${topk_pred}

# rm $checkpoint/eval_data_${pred_mode}.txt