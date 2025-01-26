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
# SBATCH --array=0-12

m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2


# sys_nopnx=/scratch/ba63/arabic-text-editing/full_taggers_new/prune_30_mt/qalb14_3
# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/nopnx_prune_20_mt/qalb14_a0.2_lbl_smthng
# sys_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20_mt/qalb14_a0.2

checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/nopnx_prune_20_mt/qalb14_a0.2/checkpoint-1500


# Checkpoints
# create_checkpoints() {
#   local sys_dir=$1
#   echo "${sys_dir}" \
#        "${sys_dir}/checkpoint-500" \
#        "${sys_dir}/checkpoint-1000" \
#        "${sys_dir}/checkpoint-1500" \
#        "${sys_dir}/checkpoint-2000" \
#        "${sys_dir}/checkpoint-2500" \
#        "${sys_dir}/checkpoint-3000" \
#        "${sys_dir}/checkpoint-3500" \
#        "${sys_dir}/checkpoint-4000" \
#        "${sys_dir}/checkpoint-4500" \
#        "${sys_dir}/checkpoint-5000" \
#        "${sys_dir}/checkpoint-5500" \
#        "${sys_dir}/checkpoint-6000"
# }


# checkpoints_nopnx=($(create_checkpoints $sys_nopnx))
# checkpoint_nopnx=${checkpoints_nopnx[$SLURM_ARRAY_TASK_ID]}


printf "Running M2 Evaluation on ${checkpoint_nopnx} ...\n"


for i in {1..3}
do
    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.${i} \
    #     --m2_file $m2_edits \


    # printf "Running M2 Evaluation without pnx on ${checkpoint_nopnx} ...\n"

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.${i}.nopnx \
    #     --m2_file $m2_edits_nopnx \


    python run_m2scorer.py \
        --system_output $checkpoint_nopnx/dev.txt.nopnx_edit.${i}.pnx_edit.pred_t05 \
        --m2_file $m2_edits \


    printf "Running M2 Evaluation without pnx on ${checkpoint_nopnx} ...\n"

    python run_m2scorer.py \
        --system_output $checkpoint_nopnx/dev.txt.nopnx_edit.${i}.pnx_edit.pred_t05.nopnx \
        --m2_file $m2_edits_nopnx \

done


