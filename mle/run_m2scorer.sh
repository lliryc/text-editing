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


# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2


m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.nopnx.m2


python /home/ba63/arabic-text-editing/tagger/run_m2scorer.py  \
    --system_output ./zaebuc_dev.mle.txt.nopnx \
    --m2_file $m2_edits_nopnx \



# python /home/ba63/arabic-text-editing/tagger/run_m2scorer.py  \
#     --system_output /scratch/ba63/arabic-text-editing/ensemble/dev.1234.nopnx.txt \
#     --m2_file $m2_edits_nopnx \




