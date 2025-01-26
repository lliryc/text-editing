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
#SBATCH --array=0-30

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=32
SEED=42
pred_mode=dev

DATA_DIR=/scratch/ba63/arabic-text-editing/tagger_data/qalb14
sys=/scratch/ba63/arabic-text-editing/full_taggers/qalb14/checkpoint-3000
LABELS=/scratch/ba63/arabic-text-editing/tagger_data/qalb14/labels.txt
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2


# Array of ranges
ranges=($(seq 0.22 0.01 0.5))

bias=${ranges[$SLURM_ARRAY_TASK_ID]}

topk_pred=.top10.${bias}

    # cp $OUTPUT_DIR/tokenizer_config.json $checkpoint
    # cp $OUTPUT_DIR/vocab.txt $checkpoint
    # cp $OUTPUT_DIR/special_tokens_map.json $checkpoint

printf "Running evaluation using ${sys} with bias ${bias}..\n"

python /home/ba63/arabic-text-editing/tagger/tag.py \
    --data_dir $DATA_DIR \
    --labels $LABELS \
    --model_name_or_path $sys \
    --output_dir $sys \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --topk_pred \
    --keep_confidence_bias $bias \
    --label_pred_output_file ${pred_mode}.preds.txt${topk_pred} \
    --rewrite_pred_output_file ${pred_mode}.txt${topk_pred} \
    --m2_edits $m2_edits \
    --m2_edits_nopnx $m2_edits_nopnx \
    --pred_mode $pred_mode # or test to get the test predictions

# Evaluation
paste $DATA_DIR/${pred_mode}.txt $sys/${pred_mode}.preds.txt${topk_pred} \
    > $sys/eval_data_${pred_mode}.txt

python /home/ba63/arabic-text-editing/tagger/evaluate.py \
    --data $sys/eval_data_${pred_mode}.txt \
    --labels $LABELS \
    --output $sys/${pred_mode}.results${topk_pred}

rm $sys/eval_data_${pred_mode}.txt
