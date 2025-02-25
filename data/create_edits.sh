#!/bin/bash
# Set number of tasks to run
#SBATCH -p compute
#SBATCH -n 2
#SBATCH -c 50
# SBATCH --mem=50GB
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# python create_edits.py \
#     --dataset qalb14-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --src_file_path  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
#     --tgt_file_path /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.cor.no_ids.dediac \
#     --create_edits \
#     --edits_granularity word \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed



# for k in 10 20 30
# do
#     python create_edits.py \
#         --dataset qalb14-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --edits_granularity subword \
#         --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed \
#         --prune \
#         --k $k \
#         --pruned_output_dir  /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed_prune_${k}
# done


# python create_edits.py \
#     --dataset qalb14+zaebuc_x20 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa \
#     --split train \
#     --edits_granularity subword \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x20/edits_compressed \
#     --prune \
#     --k 30 \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x20/edits_compressed_prune_30


# python create_edits.py \
#     --dataset qalb14_adj+zaebuc_x10-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split train \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14_adj+zaebuc_x10/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14_adj+zaebuc_x10/edits_compressed


# python create_edits.py \
#     --dataset qalb14_adj+zaebuc_x10-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split train \
#     --edits_granularity subword \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14_adj+zaebuc_x10/edits_compressed \
#     --prune \
#     --k 10 \
#     --pruned_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14_adj+zaebuc_x10/edits_compressed_prune_10


# python create_edits.py \
#     --dataset qalb14+zaebuc_x20 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa \
#     --split train \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14+zaebuc_x20/qalb14+zaebuc_x20.train.src.txt \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14+zaebuc_x20/qalb14+zaebuc_x20.train.tgt.txt \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x20/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x20/edits_compressed


# python create_edits.py \
#     --dataset zaebuc-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --src_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
#     --tgt_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.dediac \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed


# for k in 10 20 30
# do
#     python create_edits.py \
#         --dataset zaebuc-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --edits_granularity subword \
#         --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed \
#         --prune \
#         --k $k \
#         --pruned_output_dir  /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed_prune_${k}
# done




# python create_edits.py \
#     --dataset madar \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa \
#     --split test \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/coda/madar/test.preproc.raw.txt \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/coda/madar/test.preproc.coda.txt \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_compressed


# for split in train dev test
# do
#     for gran in subword word
#     do
#         for dataset in zaebuc zaebuc-arabertv02 zaebuc-arbertv2
#         do
#             if [[ "$dataset" == "zaebuc" ]]; then
#                 tokenizer=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
#             elif [[ "$dataset" == "zaebuc-arabertv02" ]]; then
#                 tokenizer=/scratch/ba63/BERT_models/bert-base-arabertv02
#             elif [[ "$dataset" == "zaebuc-arbertv2" ]]; then
#                 tokenizer=/scratch/ba63/BERT_models/ARBERTv2
#             fi

#             python create_edits.py \
#                 --dataset $dataset \
#                 --tokenizer $tokenizer \
#                 --split $split \
#                 --src_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/$split/$split.sent.raw.pnx.tok.dediac \
#                 --tgt_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/$split/$split.sent.cor.pnx.tok.dediac \
#                 --create_edits \
#                 --edits_granularity $gran \
#                 --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_no_compressed \
#                 --compress \
#                 --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/zaebuc/edits_compressed
#         done 

#     done
# done


# python create_edits.py \
#     --dataset qalb14-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split test_qalb15_L1 \
#     --src_file_path  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
#     --tgt_file_path /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.cor.no_ids.dediac \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14/edits_compressed


# python create_edits.py \
#     --dataset madar-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split dev \
#     --src_file_path  /scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.raw.txt  \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.coda.txt  \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_compressed


# for k in 10 20 30
# do
#     python create_edits.py \
#         --dataset madar-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --edits_granularity subword \
#         --compress_output_dir /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_compressed \
#         --prune \
#         --k $k \
#         --pruned_output_dir  /scratch/ba63/arabic-text-editing/edits/coda/madar/edits_compressed_prune_${k}
# done





# python create_edits.py \
#     --dataset qalb14+zaebuc_x10-arbertv2 \
#     --tokenizer /scratch/ba63/BERT_models/ARBERTv2 \
#     --split train \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14+zaebuc_x10/qalb14+zaebuc_x10_train.src.txt \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/gec/qalb14+zaebuc_x10/qalb14+zaebuc_x10_train.tgt.txt \
#     --create_edits \
#     --edits_granularity word \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed


# python create_edits.py \
#     --dataset qalb14+zaebuc_x10 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa \
#     --split dev \
#     --src_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
#     --tgt_file_path /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.cor.pnx.tok.dediac \
#     --create_edits \
#     --edits_granularity word \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/gec/qalb14+zaebuc_x10/edits_compressed


# rate=80%
# python create_edits.py \
#     --dataset madar-arabertv02 \
#     --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#     --split train \
#     --src_file_path /scratch/ba63/arabic-text-editing/data/learning-curve/madar/madar_train_${rate}.src  \
#     --tgt_file_path /scratch/ba63/arabic-text-editing/data/learning-curve/madar/madar_train_${rate}.tgt  \
#     --create_edits \
#     --edits_granularity subword \
#     --output_data_dir /scratch/ba63/arabic-text-editing/edits/learning-curve/coda/${rate}/edits_no_compressed \
#     --compress \
#     --compress_output_dir /scratch/ba63/arabic-text-editing/edits/learning-curve/coda/${rate}/edits_compressed

# for k in 10 20 30
# do
#     python create_edits.py \
#         --dataset qalb14-arabertv02 \
#         --tokenizer /scratch/ba63/BERT_models/bert-base-arabertv02 \
#         --split train \
#         --edits_granularity subword \
#         --compress_output_dir /scratch/ba63/arabic-text-editing/edits/learning-curve/gec/${rate}/edits_compressed \
#         --prune \
#         --k $k \
#         --pruned_output_dir  /scratch/ba63/arabic-text-editing/edits/learning-curve/gec/${rate}/edits_compressed_prune_${k}
# done