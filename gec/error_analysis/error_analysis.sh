#!/bin/bash

get_error_types() {
    local src=$1
    local tgt=$2
    local error_analysis_dir=$3
    local dataset=$4

    local alignment_output=${error_analysis_dir}/${dataset}.alignment.txt
    local areta_tags_output=${error_analysis_dir}/${dataset}.areta.txt
    local enriched_areta_tags_output=${error_analysis_dir}/${dataset}.areta+.txt
    local ged_output=${error_analysis_dir}/${dataset}.ged.txt
    local ged_coarse_output=${error_analysis_dir}/${dataset}.ged.coarse.txt

    printf "Generating alignments for ${src}..\n"

    python /home/ba63/gec-release/alignment/aligner.py \
        --src "${src}" \
        --tgt "${tgt}" \
        --output "${alignment_output}"

    printf "Generating areta tags for ${alignment_output}..\n"

    cd /home/ba63/gec-release/areta
    eval "$(conda shell.bash hook)"
    conda activate areta

    python annotate_err_type_ar.py \
        --alignment "${alignment_output}" \
        --output_path "${areta_tags_output}" \
        --enriched_output_path "${enriched_areta_tags_output}"

    rm -f fout2.basic

    printf "Converting areta tags to GED format..\n"

    python /home/ba63/gec-release/data/utils/create_ged_data.py \
        --input "${enriched_areta_tags_output}" \
        --output "${ged_output}"

    python /home/ba63/gec-release/data/utils/granularity_map.py \
        --input "${ged_output}" \
        --mode coarse \
        --output "${ged_coarse_output}"
}

evaluate() {
    local gold_data=$1
    local labels=$2
    local system=$3

    local eval_data="eval_data"

    paste "$gold_data" "$system" | cut -f1,2,4 > "$eval_data"

    python /home/ba63/gec-release/ged/evaluate.py \
        --data "$eval_data" \
        --labels "$labels" \
        --output "${system}.results"

    rm -f "$eval_data"
}


qalb14_dev_src=/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac
qalb14_dev_ged_gold=/home/ba63/gec-release/data/ged/qalb14/wo_camelira/coarse/dev.txt
qalb14_ged_labels=/home/ba63/gec-release/data/ged/qalb14/wo_camelira/coarse/labels.txt

zaebuc_dev_src=/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac
zaebuc_dev_ged_gold=/home/ba63/gec-release/data/ged/zaebuc/wo_camelira/coarse/dev.txt
zaebuc_ged_labels=/home/ba63/gec-release/data/ged/mix/wo_camelira/coarse/labels.txt

madar_dev_src=/scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.raw.txt
madar_dev_ged_gold=/home/ba63/text-editing-release/data/da-gec/ged/coarse/dev.txt
madar_ged_labels=/home/ba63/text-editing-release/data/da-gec/ged/coarse/labels.txt


# QALB-2014 Seq2Seq++
system_output=/scratch/ba63/backup/gec/models/gec/qalb14/full/t5_w_camelira_ged_pred_worst/checkpoint-33000/qalb14_dev.preds.txt.pp
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/t5+morph+ged-43/qalb14
get_error_types $qalb14_dev_src $system_output $error_analysis_dir qalb14_dev
evaluate $qalb14_dev_ged_gold $qalb14_ged_labels ${error_analysis_dir}/qalb14_dev.ged.coarse.txt


# ZAEBUC Seq2Seq++
system_output=/scratch/ba63/backup/gec/models/gec/mix/coarse/bart_w_camelira_ged_pred_worst/checkpoint-11000/zaebuc_dev.preds.txt
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/bart+morph+ged-13/zaebuc
get_error_types $zaebuc_dev_src $system_output $error_analysis_dir zaebuc_dev
evaluate $zaebuc_dev_ged_gold $zaebuc_ged_labels ${error_analysis_dir}/zaebuc_dev.ged.coarse.txt


#CODA Seq2Seq++
system_output=/scratch/ba63/codafication/models/t5/city/city_pred.dev.gen.txt
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/t5+city/madar
get_error_types $madar_dev_src $system_output $error_analysis_dir madar_dev
evaluate $madar_dev_ged_gold $madar_ged_labels ${error_analysis_dir}/madar_dev.ged.coarse.txt


# QALB-2014 SWEET^2_NoPnx+SWEET_Pnx
system_output=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/dev.txt.2.pnx_edit.pp
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/qalb14/sweet2_nopnx+sweet_pnx
get_error_types $qalb14_dev_src $system_output $error_analysis_dir qalb14_dev
evaluate $qalb14_dev_ged_gold $qalb14_ged_labels ${error_analysis_dir}/qalb14_dev.ged.coarse.txt


# QALB-2014 4-Ensemble
system_output=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.pp
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/qalb14/4-Ensemble
get_error_types $qalb14_dev_src $system_output $error_analysis_dir qalb14_dev
evaluate $qalb14_dev_ged_gold $qalb14_ged_labels ${error_analysis_dir}/qalb14_dev.ged.coarse.txt


# ZAEBUC SWEET^2_NoPnx+SWEET_Pnx
system_output=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-7000/dev.txt.2.pnx_edit
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/zaebuc/sweet2_nopnx+sweet_pnx
get_error_types $zaebuc_dev_src $system_output $error_analysis_dir zaebuc_dev
evaluate $zaebuc_dev_ged_gold $zaebuc_ged_labels ${error_analysis_dir}/zaebuc_dev.ged.coarse.txt


# ZAEBUC 4-Ensemble
system_output=/scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/zaebuc/4-Ensemble
get_error_types $zaebuc_dev_src $system_output $error_analysis_dir zaebuc_dev
evaluate $zaebuc_dev_ged_gold $zaebuc_ged_labels ${error_analysis_dir}/zaebuc_dev.ged.coarse.txt


# CODA SWEET
system_output=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev/arabertv02-prune10.txt
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/madar/sweet
get_error_types $madar_dev_src $system_output $error_analysis_dir madar_dev
evaluate $madar_dev_ged_gold $madar_ged_labels ${error_analysis_dir}/madar_dev.ged.coarse.txt


# CODA 4-Ensemble
system_output=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt
error_analysis_dir=/home/ba63/text-editing-release/gec/error_analysis/text-editing/madar/4-Ensemble
get_error_types $madar_dev_src $system_output $error_analysis_dir madar_dev
evaluate $madar_dev_ged_gold $madar_ged_labels ${error_analysis_dir}/madar_dev.ged.coarse.txt

