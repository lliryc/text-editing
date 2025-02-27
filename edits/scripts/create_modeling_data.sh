#!/bin/bash

# Training data
process_train_data() {
    local dataset=$1
    local base_dir=$2
    local output_dir=$3

    for comp in "compressed" "no-compressed"; do
        for level in "word-level" "subword-level"; do

            if [[ $comp == "compressed" ]]; then
                data_dir=${base_dir}/edits_compressed/${dataset}-arabertv02/${level}
            else
                data_dir=${base_dir}/edits_no_compressed/${dataset}-arabertv02/${level}
            fi

            local modeling_out="${output_dir}/${dataset}/${comp}/${level}"

            mkdir -p "${modeling_out}"

            sed 's/<s>//g' "${data_dir}/train_edits.modeling.tsv" > "${modeling_out}/train.txt"
            cut -f2 "${modeling_out}/train.txt" | sort | uniq | sed '1d' > "${modeling_out}/labels.txt"
        done
    done
}

# Training data + pruning
process_pruning() {
    local dataset=$1
    local base_dir=$2
    local output_dir=$3

    for k in 10 20 30; do
        local data_dir="${base_dir}/edits_compressed_prune_${k}/${dataset}-arabertv02/subword-level"
        local modeling_out="${output_dir}/${dataset}-prune-${k}/compressed/subword-level"

        mkdir -p "${modeling_out}"

        sed 's/<s>//g' "${data_dir}/train_edits.modeling.tsv" > "${modeling_out}/train.txt"
        cut -f2 "${modeling_out}/train.txt" | sort | uniq | sed '1d' > "${modeling_out}/labels.txt"
    done
}

# Training data Pnx Sep
process_pnx_sep() {
    local dataset=$1
    local base_dir=$2
    local output_dir=$3

    for k in 0 10 20 30; do
        if [[ $k == "0" ]]; then
            data_dir=${base_dir}/edits_compressed_pnx_sep/${dataset}-arabertv02/subword-level
            pnx_modeling_out=${output_dir}/pnx_sep/${dataset}-pnx/compressed/subword-level
            nopnx_modeling_out=${output_dir}/pnx_sep/${dataset}-nopnx/compressed/subword-level
        else
            data_dir=${base_dir}/edits_compressed_pnx_sep_prune_${k}/${dataset}-arabertv02/subword-level
            pnx_modeling_out=${output_dir}/pnx_sep/${dataset}-pnx-prune-${k}/compressed/subword-level
            nopnx_modeling_out=${output_dir}/pnx_sep/${dataset}-nopnx-prune-${k}/compressed/subword-level
        fi

        mkdir -p "${pnx_modeling_out}" "${nopnx_modeling_out}"

        sed 's/<s>//g' "${data_dir}/train_edits_pnx_edits.modeling.tsv" > "${pnx_modeling_out}/train.txt"
        cut -f2 "${pnx_modeling_out}/train.txt" | sort | uniq | sed '1d' > "${pnx_modeling_out}/labels.txt"

        sed 's/<s>//g' "${data_dir}/train_edits_nopnx_edits.modeling.tsv" > "${nopnx_modeling_out}/train.txt"
        cut -f2 "${nopnx_modeling_out}/train.txt" | sort | uniq | sed '1d' > "${nopnx_modeling_out}/labels.txt"
    done
}

# Dev/Test data (word-level and subword-level)
process_dev_test_data() {
    local dataset=$1
    local base_dir=$2
    local output_dir=$3
    local split=$4  # "dev" or "test"

    for level in "word-level" "subword-level"; do
        local data_dir="${base_dir}/edits_compressed/${dataset}-arabertv02/${level}"
        local modeling_out="${output_dir}/${dataset}"

        mkdir -p "${modeling_out}"

        if [[ "$level" == "word-level" && "$split" == "dev" ]]; then
            sed 's/<s>//g' "${data_dir}/${split}.raw.txt" > "${modeling_out}/${split}.raw.word_level.txt"
            cat "${data_dir}/${split}_edits.modeling.tsv" | cut -f1 | sed 's/<s>//g'  > "${modeling_out}/${split}.word_level.txt"
        else
            sed 's/<s>//g' "${data_dir}/${split}.raw.txt" > "${modeling_out}/${split}.raw.txt"
            cat "${data_dir}/${split}_edits.modeling.tsv" | cut -f1 | sed 's/<s>//g'  > "${modeling_out}/${split}.txt"
        fi
    done
}

# Dev data Pnx Sep
process_pnx_sep_dev() {
    local dataset=$1
    local base_dir=$2
    local output_dir=$3
    local split=$4  # "dev"

    local data_dir="${base_dir}/edits_compressed_pnx_sep/${dataset}-arabertv02/subword-level"
    local pnx_modeling_out="${output_dir}/pnx_sep/${dataset}-pnx"
    local nopnx_modeling_out="${output_dir}/pnx_sep/${dataset}-nopnx"

    mkdir -p "${pnx_modeling_out}" "${nopnx_modeling_out}"

    sed 's/<s>//g' "${data_dir}/${split}_edits_pnx.raw.txt" > "${pnx_modeling_out}/${split}.raw.txt"
    cat  "${data_dir}/${split}_edits_pnx_edits.modeling.tsv" | cut -f1 | sed 's/<s>//g' > "${pnx_modeling_out}/${split}.txt"

    sed 's/<s>//g' "${data_dir}/${split}_edits_nopnx.raw.txt" > "${nopnx_modeling_out}/${split}.raw.txt"
    cat  "${data_dir}/${split}_edits_nopnx_edits.modeling.tsv" | cut -f1 | sed 's/<s>//g' > "${nopnx_modeling_out}/${split}.txt"
}

# Datasets
train_datasets=("qalb14" "qalb14+zaebuc_x10" "zaebuc" "madar")
dev_datasets=("qalb14" "zaebuc" "madar")
test_datasets=("qalb14" "qalb15" "zaebuc" "madar")

# Process training data
for dataset in "${train_datasets[@]}"; do
    echo "Processing $dataset train..."
    if [[ "$dataset" == "madar" ]]; then
        base_dir="../data/da-gec/edits/${dataset}"
        output_dir="../data/da-gec/modeling/${dataset}"
    else
        base_dir="../data/msa-gec/edits/${dataset}"
        output_dir="../data/msa-gec/modeling/${dataset}"
    fi

    process_train_data "$dataset" "$base_dir" "$output_dir"
    process_pruning "$dataset" "$base_dir" "$output_dir"

    if [[ "$dataset" != "madar" ]]; then
        process_pnx_sep "$dataset" "$base_dir" "$output_dir"
    fi
done

# Process development data
for dataset in "${dev_datasets[@]}"; do
    echo "Processing $dataset dev..."
    if [[ "$dataset" == "madar" ]]; then
        base_dir="../data/da-gec/edits/${dataset}"
        output_dir="../data/da-gec/modeling/${dataset}"
    else
        base_dir="../data/msa-gec/edits/${dataset}"
        output_dir="../data/msa-gec/modeling/${dataset}"
    fi

    process_dev_test_data "$dataset" "$base_dir" "$output_dir" "dev"

    if [[ "$dataset" != "madar" ]]; then
        process_pnx_sep_dev "$dataset" "$base_dir" "$output_dir" "dev"
    fi
done

# Process test data
for dataset in "${test_datasets[@]}"; do
    echo "Processing $dataset test..."
    if [[ "$dataset" == "madar" ]]; then
        base_dir="../data/da-gec/edits/${dataset}"
        output_dir="../data/da-gec/modeling/${dataset}"
    else
        base_dir="../data/msa-gec/edits/${dataset}"
        output_dir="../data/msa-gec/modeling/${dataset}"
    fi

    process_dev_test_data "$dataset" "$base_dir" "$output_dir" "test"
done
