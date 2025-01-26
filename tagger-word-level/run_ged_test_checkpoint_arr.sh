#!/usr/bin/env bash
#SBATCH -p nvidia
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

BATCH_SIZE=32
SEED=42
pred_mode=dev

DATA_DIR=/scratch/ba63/arabic-text-editing/tagger_data/ged/binary/qalb14
sys=/scratch/ba63/arabic-text-editing/ged/binary/qalb14
LABELS=/scratch/ba63/arabic-text-editing/tagger_data/ged/binary/qalb14/labels.txt



# Array of checkpoints
checkpoints=(
  "${sys}"
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

python /home/ba63/arabic-text-editing/tagger/ged.py \
    --data_dir $DATA_DIR \
    --labels $LABELS \
    --model_name_or_path $checkpoint \
    --output_dir $checkpoint \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt \
    --pred_mode $pred_mode # or test to get the test predictions

# Evaluation
paste $DATA_DIR/${pred_mode}.txt $checkpoint/${pred_mode}.preds.txt \
    > $checkpoint/eval_data_${pred_mode}.txt

python /home/ba63/arabic-text-editing/tagger/eval_ged.py \
    --data $checkpoint/eval_data_${pred_mode}.txt \
    --labels $LABELS \
    --output $checkpoint/${pred_mode}.results

rm $checkpoint/eval_data_${pred_mode}.txt
