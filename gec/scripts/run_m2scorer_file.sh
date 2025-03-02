#!/bin/bash
# Set number of tasks to run
# SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err



# m2_edits=/home/ba63/text-editing/data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/qalb14/qalb14_dev.nopnx.m2

# m2_edits=/home/ba63/text-editing/data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.m2
# m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/qalb14/qalb14_test.nopnx.m2

# m2_edits=/home/ba63/text-editing/data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.m2
# m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/qalb15/qalb15_L1-test.nopnx.m2

# m2_edits=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_dev.m2
# m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_dev.nopnx.m2

# m2_edits=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_test.m2
# m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_test.nopnx.m2


# sys=predictions/ensembles/zaebuc

# python utils/run_m2scorer.py \
#     --system_output $sys/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt \
#     --m2_file $m2_edits

# python utils/run_m2scorer.py \
#     --system_output $sys/test.ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.nopnx \
#     --m2_file $m2_edits_nopnx


m2_edits=/home/ba63/text-editing/data/da-gec/m2edits/test.m2

sys=predictions/ensembles/madar

python utils/run_m2scorer.py \
    --system_output $sys/test.ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt \
    --m2_file $m2_edits





