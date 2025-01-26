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

test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep/pnx/qalb14/dev_pnx.txt

checkpoint=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20/qalb14/checkpoint-avg
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep_prune_20/pnx/qalb14/labels.txt


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

