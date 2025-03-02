#!/bin/bash
# Set number of tasks to run
# SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.%A.out
#SBATCH -e job.%J.%A.err
#SBATCH --array=0-4



# m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2

# m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_test.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_test.nopnx.m2

m2_edits=/home/ba63/codafication/data/m2-files/dev.m2

# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers_arbertv2/qalb14-word-level-no-compressed-a100
# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-pnx-prune-30-a100

# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x20/pnx_taggers_arabertv02/qalb14+zaebuc_x20-nopnx-prune-30-a100
# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-pnx-prune-10-a100
# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers_arbertv2_15_iter/qalb14+zaebuc_x10-no-compressed-a100


sys_nopnx=/scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arbertv2/madar-no-compressed-a100


# Checkpoints
create_checkpoints() {
  local sys_dir=$1
  echo "${sys_dir}" \
    "${sys_dir}/checkpoint-500" \
    "${sys_dir}/checkpoint-1000" \
    "${sys_dir}/checkpoint-1500" \
    "${sys_dir}/checkpoint-2000"
}

# Checkpoints
# create_checkpoints() {
#   local sys_dir=$1
#   echo "${sys_dir}" \
#     "${sys_dir}/checkpoint-500" \
#     "${sys_dir}/checkpoint-1000" \
#     "${sys_dir}/checkpoint-1500" \
#     "${sys_dir}/checkpoint-2000" \
#     "${sys_dir}/checkpoint-2500" \
#     "${sys_dir}/checkpoint-3000" \
#     "${sys_dir}/checkpoint-3500" \
#     "${sys_dir}/checkpoint-4000" \
#     "${sys_dir}/checkpoint-4500" \
#     "${sys_dir}/checkpoint-5000" \
#     "${sys_dir}/checkpoint-5500" \
#     "${sys_dir}/checkpoint-6000"
# }


# Checkpoints
# create_checkpoints() {
#   local sys_dir=$1
#   echo "${sys_dir}" \
#     "${sys_dir}/checkpoint-500" \
#     "${sys_dir}/checkpoint-1000" \
#     "${sys_dir}/checkpoint-1500" \
#     "${sys_dir}/checkpoint-2000" \
#     "${sys_dir}/checkpoint-2500" \
#     "${sys_dir}/checkpoint-3000" \
#     "${sys_dir}/checkpoint-3500" \
#     "${sys_dir}/checkpoint-4000" \
#     "${sys_dir}/checkpoint-4500" \
#     "${sys_dir}/checkpoint-5000" \
#     "${sys_dir}/checkpoint-5500" \
#     "${sys_dir}/checkpoint-6000" \
#     "${sys_dir}/checkpoint-6500" \
#     "${sys_dir}/checkpoint-7000"  \
#     "${sys_dir}/checkpoint-7500" \
#     "${sys_dir}/checkpoint-8000" \
#     "${sys_dir}/checkpoint-8500"  \
#     "${sys_dir}/checkpoint-9000" \
#     "${sys_dir}/checkpoint-9500"
# }


checkpoints_nopnx=($(create_checkpoints $sys_nopnx))
checkpoint_nopnx=${checkpoints_nopnx[$SLURM_ARRAY_TASK_ID]}

for i in {1..3}
do
    python utils/run_m2scorer.py \
        --system_output $checkpoint_nopnx/dev.txt.${i} \
        --m2_file $m2_edits

    # python utils/run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/test.txt.${i}.nopnx \
    #     --m2_file $m2_edits_nopnx

done