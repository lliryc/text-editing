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


m2_edits=/home/ba63/text-editing/data/msa-gec/raw/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/qalb14/qalb14_dev.nopnx.m2

sys=predictions/oracle/qalb14

for comp in compressed no_compressed
do
    for gran in word subword
    do
        python utils/run_m2scorer.py \
            --system_output $sys/qalb14_dev_${gran}_${comp}.txt \
            --m2_file $m2_edits
    done
done


for k in 10 20 30
do
    python utils/run_m2scorer.py \
        --system_output $sys/prune/qalb14_dev_subword_compressed_prune_${k}.txt \
        --m2_file $m2_edits

    python  utils/run_m2scorer.py \
        --system_output $sys/pnx_sep/qalb14_dev_subword_compressed_prune_${k}_pnx.txt  \
        --m2_file $m2_edits

    python  utils/run_m2scorer.py \
        --system_output $sys/pnx_sep/qalb14_dev_subword_compressed_prune_${k}_nopnx.txt  \
        --m2_file $m2_edits_nopnx
done

python  utils/run_m2scorer.py \
    --system_output $sys/pnx_sep/qalb14_dev_subword_compressed_pnx.txt  \
    --m2_file $m2_edits

python  utils/run_m2scorer.py \
    --system_output $sys/pnx_sep/qalb14_dev_subword_compressed_nopnx.txt  \
    --m2_file $m2_edits_nopnx




m2_edits=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_dev.m2
m2_edits_nopnx=/home/ba63/text-editing/data/msa-gec/m2edits/zaebuc/zaebuc_dev.nopnx.m2

sys=predictions/oracle/zaebuc

for comp in compressed no_compressed
do
    for gran in word subword
    do
        python utils/run_m2scorer.py \
            --system_output $sys/zaebuc_dev_${gran}_${comp}.txt \
            --m2_file $m2_edits
    done
done


for k in 10 20 30
do
    python utils/run_m2scorer.py \
        --system_output $sys/prune/zaebuc_dev_subword_compressed_prune_${k}.txt \
        --m2_file $m2_edits

    python  utils/run_m2scorer.py \
        --system_output $sys/pnx_sep/zaebuc_dev_subword_compressed_prune_${k}_pnx.txt  \
        --m2_file $m2_edits

    python  utils/run_m2scorer.py \
        --system_output $sys/pnx_sep/zaebuc_dev_subword_compressed_prune_${k}_nopnx.txt  \
        --m2_file $m2_edits_nopnx
done

python  utils/run_m2scorer.py \
    --system_output $sys/pnx_sep/zaebuc_dev_subword_compressed_pnx.txt  \
    --m2_file $m2_edits

python  utils/run_m2scorer.py \
    --system_output $sys/pnx_sep/zaebuc_dev_subword_compressed_nopnx.txt  \
    --m2_file $m2_edits_nopnx



m2_edits=/home/ba63/text-editing/data/da-gec/m2edits/dev.m2
sys=predictions/oracle/madar

for comp in compressed no_compressed
do
    for gran in word subword
    do
        python utils/run_m2scorer.py \
            --system_output $sys/madar_dev_${gran}_${comp}.txt \
            --m2_file $m2_edits
    done
done


for k in 10 20 30
do
    python utils/run_m2scorer.py \
        --system_output $sys/prune/madar_dev_subword_compressed_prune_${k}.txt \
        --m2_file $m2_edits

done


