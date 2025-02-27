#!/bin/bash
# Set number of tasks to run
#SBATCH -p compute
#SBATCH -n 2
#SBATCH -c 75
# SBATCH --mem=50GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# Train Edits
declare -A src_paths_train=(
    ["qalb14"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac"
    ["zaebuc"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/train/train.sent.raw.pnx.tok.dediac"
    ["qalb14+zaebuc_x10"]="../data/msa-gec/raw/qalb14+zaebuc_x10/train.src.txt"
    ["madar"]="../data/da-gec/raw/train.preproc.raw.txt"
)

declare -A tgt_paths_train=(
    ["qalb14"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.cor.no_ids.dediac"
    ["zaebuc"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/train/train.sent.cor.pnx.tok.dediac"
    ["qalb14+zaebuc_x10"]="../data/msa-gec/raw/qalb14+zaebuc_x10/train.tgt.txt"
    ["madar"]="../data/da-gec/raw/train.preproc.coda.txt"
)


datasets=("qalb14" "zaebuc" "qalb14+zaebuc_x10" "madar")

# Process each dataset
for dataset in "${datasets[@]}"; do
    src="${src_paths_train[$dataset]}"
    tgt="${tgt_paths_train[$dataset]}"

    if [[ "$dataset" == "madar" ]]; then
        base_dir=../data/da-gec/edits/${dataset}
    else
        base_dir=../data/msa-gec/edits/${dataset}
    fi

    output_dir=${base_dir}/edits_no_compressed
    output_dir_comp=${base_dir}/edits_compressed
    
    for gran in word subword; do
        echo "Creating edits for ${dataset} at the ${gran} level..."
        python create_edits.py \
            --dataset ${dataset}-arabertv02 \
            --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
            --split train \
            --src_file_path ${src} \
            --tgt_file_path ${tgt} \
            --create_edits \
            --edits_granularity ${gran} \
            --output_data_dir ${output_dir} \
            --compress \
            --compress_output_dir ${output_dir_comp}
    done

    # Pruning
    for k in 10 20 30; do
        echo "Pruning edits that appear <= ${k} times from ${dataset}..."
        output_dir_prune=${base_dir}/edits_compressed_prune_${k}
        python create_edits.py \
            --dataset ${dataset}-arabertv02 \
            --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
            --split train \
            --edits_granularity subword \
            --compress_output_dir ${output_dir_comp} \
            --prune \
            --k $k \
            --pruned_output_dir ${output_dir_prune}
    done

done


# Dev/Test Edits
declare -A paths=(
    ["dev_qalb14_src"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac"
    ["dev_qalb14_tgt"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac"
    
    ["dev_zaebuc_src"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac"
    ["dev_zaebuc_tgt"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.dediac"
    
    ["dev_madar_src"]="../data/da-gec/raw/dev.preproc.raw.txt"
    ["dev_madar_tgt"]="../data/da-gec/raw/dev.preproc.coda.txt"

    ["test_qalb14_src"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac"
    ["test_qalb14_tgt"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.cor.no_ids.dediac"

    ["test_qalb15_src"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac"
    ["test_qalb15_tgt"]="../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids.dediac"

    ["test_zaebuc_src"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac"
    ["test_zaebuc_tgt"]="../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/test/test.sent.cor.pnx.tok.dediac"

    ["test_madar_src"]="../data/da-gec/raw/test.preproc.raw.txt"
    ["test_madar_tgt"]="../data/da-gec/raw/test.preproc.coda.txt"
)


# Function to process datasets
process_dataset() {
    local split=$1  # dev or test
    shift
    local datasets=("$@")  # Remaining arguments are dataset names

    for dataset in "${datasets[@]}"; do
        src="${paths[${split}_${dataset}_src]}"
        tgt="${paths[${split}_${dataset}_tgt]}"
        
        if [[ "$dataset" == "madar" ]]; then
            base_dir=../data/da-gec/edits/${dataset}
        else
            base_dir=../data/msa-gec/edits/${dataset}
        fi
        
        output_dir=${base_dir}/edits_no_compressed
        output_dir_comp=${base_dir}/edits_compressed

        for gran in word subword; do
            echo "Creating edits for ${dataset} ${split} at the ${gran} level..."
            python create_edits.py \
                --dataset ${dataset}-arabertv02 \
                --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
                --split ${split} \
                --src_file_path ${src} \
                --tgt_file_path ${tgt} \
                --create_edits \
                --edits_granularity ${gran} \
                --output_data_dir ${output_dir} \
                --compress \
                --compress_output_dir ${output_dir_comp}
        done
    done
}

# Run for both dev and test splits
process_dataset "dev" "qalb14" "zaebuc" "madar"
process_dataset "test" "qalb14" "qalb15" "zaebuc" "madar"