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

outputs_dir=predictions/ensembles/zaebuc

python ensemble.py \
    --input_file ../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac \
    --models_outputs \
        predictions/seq2seq/zaebuc/dev.arabart+morph+ged-13.txt \
        predictions/taggers/arabertv02/zaebuc/qalb14+zaebuc_x10-prune-30/compressed/subword-level/dev.txt.2 \
        predictions/taggers/arabertv02/zaebuc/pnx_seg/qalb14+zaebuc_x10-nopnx-prune-30+pnx-prune-10/compressed/subword-level/dev.txt.2.pnx_edit \
    --voting_threshold 2 \
    --task msa-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt


python ensemble.py \
    --input_file  ../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac  \
    --models_outputs \
        predictions/seq2seq/zaebuc/dev.arabart+morph+ged-13.txt \
        predictions/taggers/arabertv02/zaebuc/qalb14+zaebuc_x10-prune-30/compressed/subword-level/dev.txt.2 \
        predictions/taggers/arabertv02/zaebuc/pnx_seg/qalb14+zaebuc_x10-nopnx-prune-30+pnx-prune-10/compressed/subword-level/dev.txt.2.pnx_edit \
        predictions/llms/gpt-4o/5-shot-en/zaebuc/dev.preds.txt \
    --voting_threshold 3 \
    --task msa-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=predictions/ensembles/qalb14

python ensemble.py \
    --input_file  ../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
    --models_outputs  \
        predictions/seq2seq/qalb14/dev.arat5+morph+ged-43.txt \
        predictions/taggers/arabertv02/qalb14/qalb14-prune-10/compressed/subword-level/dev.txt.2 \
        predictions/taggers/arabertv02/qalb14/pnx_seg/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/dev.txt.2.pnx_edit \
    --voting_threshold 2 \
    --task msa-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac \
    --models_outputs  \
        predictions/seq2seq/qalb14/dev.arat5+morph+ged-43.txt \
        predictions/taggers/arabertv02/qalb14/qalb14-prune-10/compressed/subword-level/dev.txt.2 \
        predictions/taggers/arabertv02/qalb14/pnx_seg/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/dev.txt.2.pnx_edit \
        predictions/llms/gpt-4o/5-shot-en/qalb14/dev.preds.txt \
    --voting_threshold 3 \
    --task msa-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=predictions/ensembles/madar

python ensemble.py \
    --input_file  ../data/da-gec/raw/dev.preproc.raw.txt \
    --models_outputs  \
        predictions/seq2seq/madar/dev.t5+city.txt \
        predictions/taggers/camelbert-msa/madar/madar-prune-10/compressed/subword-level/dev.txt.1 \
        predictions/taggers/arabertv02/madar/madar-prune-10/compressed/subword-level/dev.txt.1 \
    --voting_threshold 2 \
    --task da-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt


python ensemble.py \
    --input_file  /scratch/ba63/arabic-text-editing/data/coda/madar/dev.preproc.raw.txt \
    --models_outputs  \
        predictions/seq2seq/madar/dev.t5+city.txt \
        predictions/taggers/camelbert-msa/madar/madar-prune-10/compressed/subword-level/dev.txt.1 \
        predictions/taggers/arabertv02/madar/madar-prune-10/compressed/subword-level/dev.txt.1 \
        predictions/llms/gpt-4o/5-shot-en/madar/dev.preds.txt \
    --voting_threshold 3 \
    --task da-gec \
    --output_path $outputs_dir/dev.ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt



##### TEST #####

outputs_dir=predictions/ensembles/zaebuc

python ensemble.py \
    --input_file  ../data/msa-gec/raw/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac \
    --models_outputs  \
        predictions/seq2seq/zaebuc/test.arabart+ged-13.txt \
        predictions/taggers/arabertv02/zaebuc/qalb14+zaebuc_x10-prune-30/compressed/subword-level/test.txt.2 \
        predictions/taggers/arabertv02/zaebuc/pnx_seg/qalb14+zaebuc_x10-nopnx-prune-30+pnx-prune-10/compressed/subword-level/test.txt.2.pnx_edit \
    --voting_threshold 2 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt


python ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac \
    --models_outputs  \
        predictions/seq2seq/zaebuc/test.arabart+ged-13.txt \
        predictions/taggers/arabertv02/zaebuc/qalb14+zaebuc_x10-prune-30/compressed/subword-level/test.txt.2 \
        predictions/taggers/arabertv02/zaebuc/pnx_seg/qalb14+zaebuc_x10-nopnx-prune-30+pnx-prune-10/compressed/subword-level/test.txt.2.pnx_edit \
        predictions/llms/gpt-4o/5-shot-en/zaebuc/test.preds.txt \
    --voting_threshold 3 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt



outputs_dir=predictions/ensembles/qalb14

python ensemble.py \
    --input_file  ../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac \
    --models_outputs  \
        predictions/seq2seq/qalb14/test.arabart+ged-43.txt \
        predictions/taggers/arabertv02/qalb14/qalb14-prune-10/compressed/subword-level/test.txt.2 \
        predictions/taggers/arabertv02/qalb14/pnx_seg/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/test.txt.2.pnx_edit \
    --voting_threshold 2 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python ensemble.py \
    --input_file  /home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac \
    --models_outputs  \
        predictions/seq2seq/qalb14/test.arabart+ged-43.txt \
        predictions/taggers/arabertv02/qalb14/qalb14-prune-10/compressed/subword-level/test.txt.2 \
        predictions/taggers/arabertv02/qalb14/pnx_seg/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/test.txt.2.pnx_edit \
        predictions/llms/gpt-4o/5-shot-en/qalb14/test.preds.txt \
    --voting_threshold 3 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt


outputs_dir=predictions/ensembles/qalb15

python ensemble.py \
    --input_file  ../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
    --models_outputs \
        predictions/seq2seq/qalb15/test.arabart+morph+ged-43.txt \
        predictions/taggers/arabertv02/qalb15/qalb14-prune-10/compressed/subword-level/test_qalb15_L1.txt.2 \
        predictions/taggers/arabertv02/qalb15/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/test_qalb15_L1.txt.2.pnx_edit \
    --voting_threshold 2 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx.txt

python ensemble.py \
    --input_file  ../data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac \
    --models_outputs \
        predictions/seq2seq/qalb15/test.arabart+morph+ged-43.txt \
        predictions/taggers/arabertv02/qalb15/qalb14-prune-10/compressed/subword-level/test_qalb15_L1.txt.2 \
        predictions/taggers/arabertv02/qalb15/qalb14-nopnx-prune-30+pnx-prune-20/compressed/subword-level/test_qalb15_L1.txt.2.pnx_edit \
        predictions/llms/gpt-4o/5-shot-en/qalb15/test.preds.txt \
    --voting_threshold 3 \
    --task msa-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt



outputs_dir=predictions/ensembles/madar

python ensemble.py \
    --input_file  ../data/da-gec/raw/test.preproc.raw.txt \
    --models_outputs  \
        predictions/seq2seq/madar/test.t5+da_phrase.txt \
        predictions/taggers/camelbert-msa/madar/madar-prune-10/compressed/subword-level/test.txt \
        predictions/taggers/arabertv02/madar/madar-prune-10/compressed/subword-level/test.txt \
    --voting_threshold 2 \
   --task da-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt


python ensemble.py \
    --input_file  ../data/da-gec/raw/test.preproc.raw.txt \
    --models_outputs  \
        predictions/seq2seq/madar/test.t5+da_phrase.txt \
        predictions/taggers/camelbert-msa/madar/madar-prune-10/compressed/subword-level/test.txt \
        predictions/taggers/arabertv02/madar/madar-prune-10/compressed/subword-level/test.txt \
        predictions/llms/gpt-4o/5-shot-en/madar/test.preds.txt \
    --voting_threshold 3 \
   --task da-gec \
    --output_path $outputs_dir/test.ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt