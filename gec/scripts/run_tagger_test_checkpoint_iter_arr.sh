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
#SBATCH --array=0-4

nvidia-smi
module purge

#################################################
# Tagger Test Eval Script with iterative decoding 
#################################################

BATCH_SIZE=64
SEED=42
pred_mode=dev

test_file=/home/ba63/text-editing/data/da-gec/modeling/madar/madar/dev.txt
test_file_raw=/home/ba63/text-editing/data/da-gec/modeling/madar/madar/dev.raw.txt

sys=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100
labels=/home/ba63/text-editing/data/da-gec/modeling/madar/madar-prune-10/compressed/subword-level/labels.txt

checkpoints=(
    "${sys}" 
    "${sys}/checkpoint-500" 
    "${sys}/checkpoint-1000" 
    "${sys}/checkpoint-1500" 
    "${sys}/checkpoint-2000"
)


# test_file=/home/ba63/text-editing/data/msa-gec/modeling/zaebuc/pnx_sep/zaebuc-pnx/dev.txt
# test_file_raw=/home/ba63/text-editing/data/msa-gec/modeling/zaebuc/pnx_sep/zaebuc-pnx/dev.raw.txt

# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-pnx-prune-30-a100
# labels=/home/ba63/text-editing/data/msa-gec/modeling/qalb14+zaebuc_x10/pnx_sep/qalb14+zaebuc_x10-pnx-prune-30/compressed/subword-level/labels.txt


# checkpoints=(
#     "${sys}" 
#     "${sys}/checkpoint-500" 
#     "${sys}/checkpoint-1000" 
#     "${sys}/checkpoint-1500" 
#     "${sys}/checkpoint-2000" 
#     "${sys}/checkpoint-2500" 
#     "${sys}/checkpoint-3000" 
#     "${sys}/checkpoint-3500" 
#     "${sys}/checkpoint-4000"
#     "${sys}/checkpoint-4500"
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

# cp $sys/tokenizer_config.json $checkpoint
# cp $sys/vocab.txt $checkpoint
# cp $sys/special_tokens_map.json $checkpoint

printf "Running evaluation using ${checkpoint}..\n"

# iterative prediction
for i in {1..3}
do
    if [[ $i != 1 ]]; then
        test_file=$checkpoint/${pred_mode}.txt.$((i - 1)).tokens
        test_file_raw=$checkpoint/${pred_mode}.txt.$((i - 1)).raw.tokens
        echo $test_file
    fi

    python tag.py \
        --tokenized_data_path $test_file \
        --tokenized_raw_data_path $test_file_raw \
        --labels $labels \
        --input_unit subword-level \
        --task msa-gec \
        --model_name_or_path $checkpoint \
        --output_dir $checkpoint \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed $SEED \
        --do_pred \
        --label_pred_output_file ${pred_mode}.preds.txt.${i} \
        --rewrite_pred_output_file ${pred_mode}.txt.${i}

    python utils/tokenize_data.py \
        --input $checkpoint/${pred_mode}.txt.${i}  \
        --tokenizer_path $checkpoint

done