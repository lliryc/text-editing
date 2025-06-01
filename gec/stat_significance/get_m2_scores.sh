#!/bin/bash
# Set number of tasks to run
#SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

m2_scorer=/home/ba63/text-editing-release/gec/utils/run_m2scorer.py

## QALB-2014 Dev
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2

# sweet^2_nopnx + sweet_pnx
# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/dev.txt.2.pnx_edit.pp

# printf "Evaluating ${sys}\n"

# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.pp
# printf "Evaluating ${sys}\n"

# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

## QALB-2014 Test
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2
# sweet^2_nopnx + sweet_pnx
# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/test.txt.2.pnx_edit
# printf "Evaluating ${sys}\n"

# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt
# printf "Evaluating ${sys}\n"

# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

## QALB-2015-L1 Test
# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.m2
# sweet^2_nopnx + sweet_pnx
# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/test_qalb15_L1.txt.2.pnx_edit
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test_qalb15_L1/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single


# ZAEBUC Dev
# m2_edits=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_dev.m2
# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# ZAEBUC Test
# m2_edits=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_test.m2
# sweet^2_nopnx + sweet_pnx
# sys=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-7000/test.txt.2.pnx_edit
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single


# MADAR CODA Dev
# m2_edits=/home/ba63/text-editing/data/da-gec/m2edits/dev.m2
# SWEET
# sys=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/dev.txt.1
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
# sys=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt
# printf "Evaluating ${sys}\n"
# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single


# MADAR CODA Test  
m2_edits=/home/ba63/text-editing/data/da-gec/m2edits/test.m2
# SWEET
# sys=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/test.txt
# printf "Evaluating ${sys}\n"

# python $m2_scorer \
#     --system_output $sys \
#     --m2_file $m2_edits \
#     --mode single

# 4-Ensemble
sys=/scratch/ba63/arabic-text-editing/ensemble/coda/madar/test/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt
python $m2_scorer \
    --system_output $sys \
    --m2_file $m2_edits \
    --mode single


