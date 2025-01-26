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
#SBATCH --array=0-12

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=dev
var=qalb14_3


test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_word_level/qalb14/no-compressed/dev.txt


# test_file=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/pnx/${var}/dev_pnx.txt
# test_file=/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/nopnx/${var}/dev_nopnx.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-7500/dev.txt.3.tokens
# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/pnx_sep/pnx/zaebuc/dev_pnx.txt
# test_file=/scratch/ba63/arabic-text-editing/coda_data/madar-arabertv02-large/madar/dev.txt

# test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/zaebuc/dev.txt

sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers/qalb14-word-level-no-compressed
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_word_level/qalb14/no-compressed/labels.txt


# test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data+morph/qalb14/dev.txt

# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers+morph/pnx_sep_prune_30_cor/nopnx/${var}
# labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data+morph/pnx_sep_prune_30_cor/nopnx/${var}/labels.txt

# test_file=/scratch/ba63/arabic-text-editing/coda_data/madar/madar/dev.txt
# sys=/scratch/ba63/arabic-text-editing/coda_taggers/taggers/madar-mix
# labels=/scratch/ba63/arabic-text-editing/coda_data/madar/madar/labels.txt



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

# checkpoints=(
#     "${sys}"
#     "${sys}/checkpoint-1000"
#     "${sys}/checkpoint-10000"
#     "${sys}/checkpoint-10500"
#     "${sys}/checkpoint-11000"
#     "${sys}/checkpoint-11500"
#     "${sys}/checkpoint-12000"
#     "${sys}/checkpoint-12500"
#     "${sys}/checkpoint-13000"
#     "${sys}/checkpoint-1500"
#     "${sys}/checkpoint-2000"
#     "${sys}/checkpoint-2500"
#     "${sys}/checkpoint-3000"
#     "${sys}/checkpoint-3500"
#     "${sys}/checkpoint-4000"
#     "${sys}/checkpoint-4500"
#     "${sys}/checkpoint-500"
#     "${sys}/checkpoint-5000"
#     "${sys}/checkpoint-5500"
#     "${sys}/checkpoint-6000"
#     "${sys}/checkpoint-6500"
#     "${sys}/checkpoint-7000"
#     "${sys}/checkpoint-7500"
#     "${sys}/checkpoint-8000"
#     "${sys}/checkpoint-8500"
#     "${sys}/checkpoint-9000"
#     "${sys}/checkpoint-9500"
# )


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

    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --file_path $test_file \
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
