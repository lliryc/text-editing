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
#SBATCH --array=0-19

nvidia-smi
module purge

#################################
# Tagger TEST EVAL SCRIPT
#################################

BATCH_SIZE=64
SEED=42
pred_mode=dev

# test_file=/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data_word_level_arbertv2/qalb14/no-compressed/dev.txt
test_file=/scratch/ba63/arabic-text-editing/gec_data/zaebuc/tagger_data_word_level_arbertv2/zaebuc/dev.txt

sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers_arbertv2_15_iter/qalb14+zaebuc_x10-word-level-no-compressed-a100
labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_word_level_arbertv2/no-compressed/labels.txt


# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers/qalb14+zaebuc_x10-word-level-a100
# labels=/scratch/ba63/arabic-text-editing/gec_data/qalb14+zaebuc_x10/tagger_data_word_level/compressed/labels.txt


# test_file=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data_word_level/madar/compressed/dev.txt

# sys=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers/madar-word-level-a100
# labels=/scratch/ba63/arabic-text-editing/coda_data/madar/tagger_data_word_level/madar/compressed/labels.txt



# Array of checkpoints
# checkpoints=(
#   "${sys}"
#   "${sys}/checkpoint-500"
#   "${sys}/checkpoint-1000"
#   "${sys}/checkpoint-1500"
#   "${sys}/checkpoint-2000"
# )


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
    "${sys}/checkpoint-6500" 
    "${sys}/checkpoint-7000"  
    "${sys}/checkpoint-7500" 
    "${sys}/checkpoint-8000" 
    "${sys}/checkpoint-8500"  
    "${sys}/checkpoint-9000" 
    "${sys}/checkpoint-9500"
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

    python tag.py \
        --file_path $test_file \
        --labels $labels \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

        awk '{for (i=1; i<=NF; i++) print $i; print ""}' $checkpoint/${pred_mode}.txt.${i} > $checkpoint/${pred_mode}.txt.${i}.tokens

done