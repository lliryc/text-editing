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


##### DEV #####

outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/dev

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
    --models_outputs  $outputs_dir/arabart+morph+ged13.txt $outputs_dir/arabertv02_prune30.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune10.txt \
    --voting_threshold 2 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt


python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
    --models_outputs  $outputs_dir/arabart+morph+ged13.txt $outputs_dir/arabertv02_prune30.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune10.txt $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/dev

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
    --models_outputs  $outputs_dir/arat5+morph+ged-43.txt $outputs_dir/arabertv02_prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt   \
    --voting_threshold 2 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
    --models_outputs  $outputs_dir/arat5+morph+ged-43.txt $outputs_dir/arabertv02_prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt  $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev

python edits_ensemble.py \
    --input_file  /scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.raw.txt \
    --models_outputs  $outputs_dir/t5+city.txt $outputs_dir/camelbert-prune10.txt  $outputs_dir/arabertv02-prune10.txt \
    --voting_threshold 2 \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt


python edits_ensemble.py \
    --input_file  /scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.raw.txt \
    --models_outputs  $outputs_dir/t5+city.txt $outputs_dir/camelbert-prune10.txt  $outputs_dir/arabertv02-prune10.txt $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt



##### TEST #####

outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/test

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac \
    --models_outputs  $outputs_dir/arabart+ged13.txt $outputs_dir/arabertv02_prune30.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune10.txt \
    --voting_threshold 2 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt


python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac \
    --models_outputs  $outputs_dir/arabart+ged13.txt $outputs_dir/arabertv02_prune30.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune10.txt $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac \
    --models_outputs  $outputs_dir/arabart+ged-43.txt $outputs_dir/arabertv02-prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt   \
    --voting_threshold 2 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac \
    --models_outputs  $outputs_dir/arabart+ged-43.txt $outputs_dir/arabertv02-prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt  $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test_qalb15_L1

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
    --models_outputs  $outputs_dir/arabart+morph+ged-32.txt $outputs_dir/arabertv02-prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt   \
    --voting_threshold 2 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python edits_ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
    --models_outputs  $outputs_dir/arabart+morph+ged-32.txt $outputs_dir/arabertv02-prune10.txt $outputs_dir/arabertv02_nopnx-prune30_pnx-prune20.txt  $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --pnx_proc \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt



outputs_dir=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/test

python edits_ensemble.py \
    --input_file  /scratch/ba63/arabic-text-editing/data/coda/madar/test.preproc.raw.txt \
    --models_outputs  $outputs_dir/t5+da_phrase.txt $outputs_dir/camelbert-prune10.txt  $outputs_dir/arabertv02-prune10.txt \
    --voting_threshold 2 \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt


python edits_ensemble.py \
    --input_file  /scratch/ba63/arabic-text-editing/data/coda/madar/test.preproc.raw.txt \
    --models_outputs  $outputs_dir/t5+da_phrase.txt $outputs_dir/camelbert-prune10.txt  $outputs_dir/arabertv02-prune10.txt $outputs_dir/gpt4o.txt \
    --voting_threshold 3 \
    --output_path $outputs_dir/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt