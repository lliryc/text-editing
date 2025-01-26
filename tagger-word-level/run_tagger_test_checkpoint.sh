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

BATCH_SIZE=32
SEED=42
pred_mode=dev

DATA_DIR=/scratch/ba63/arabic-text-editing/tagger_data/qalb14
OUTPUT_DIR=/scratch/ba63/arabic-text-editing/full_taggers/qalb14
LABELS=/scratch/ba63/arabic-text-editing/tagger_data/qalb14/labels.txt
m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2
topk_pred='.top10.2conf'

for checkpoint in $OUTPUT_DIR/checkpoint-* $OUTPUT_DIR 

do
    # cp $OUTPUT_DIR/tokenizer_config.json $checkpoint
    # cp $OUTPUT_DIR/vocab.txt $checkpoint
    # cp $OUTPUT_DIR/special_tokens_map.json $checkpoint

    printf "Running evaluation using ${checkpoint}..\n"

    python /home/ba63/arabic-text-editing/tagger/tag.py \
        --data_dir $DATA_DIR \
        --labels $LABELS \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --topk_pred \
        --keep_confidence_bias 0.1 \
        --label_pred_output_file ${pred_mode}.preds.txt${topk_pred} \
        --rewrite_pred_output_file ${pred_mode}.txt${topk_pred} \
        --m2_edits $m2_edits \
        --m2_edits_nopnx $m2_edits_nopnx \
        --pred_mode $pred_mode # or test to get the test predictions

    # Evaluation
    paste $DATA_DIR/${pred_mode}.txt $checkpoint/${pred_mode}.preds.txt${topk_pred} \
        > $checkpoint/eval_data_${pred_mode}.txt

    python /home/ba63/arabic-text-editing/tagger/evaluate.py \
        --data $checkpoint/eval_data_${pred_mode}.txt \
        --labels $LABELS \
        --output $checkpoint/${pred_mode}.results${topk_pred}

    rm $checkpoint/eval_data_${pred_mode}.txt


done



