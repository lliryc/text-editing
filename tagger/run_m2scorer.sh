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
#SBATCH --array=0-12



m2_edits=/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.m2
m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/qalb14/qalb14_dev.nopnx.m2

# m2_edits=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.m2
# m2_edits_nopnx=/home/ba63/gec-release/data/m2edits/zaebuc/zaebuc_dev.nopnx.m2


# m2_edits=/home/ba63/codafication/data/m2-files/dev.m2


# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_20/qalb14/checkpoint-1500
# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/pnx_taggers/nopnx_prune_30_s1234/qalb14/checkpoint-3000

# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_20/qalb14/checkpoint-avg

# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers+morph/pnx_sep_prune_30_cor/nopnx/qalb14_3
# sys_pnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers/pnx_prune_30_cor/qalb14_3


sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14/taggers/qalb14-no-compressed-v100


# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14_adj+zaebuc_x10/pnx_taggers/pnx_prune_30
# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14_adj+zaebuc_x10/pnx_taggers/nopnx_prune_20/checkpoint-5500

# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/ensemble

# checkpoint_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_20_iters/nopnx/checkpoint-7500
# sys_nopnx=/scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/taggers_20_iters/qalb14+zaebuc_x10_prune_30

# Checkpoints
create_checkpoints() {
  local sys_dir=$1
  echo "${sys_dir}" \
    "${sys_dir}/checkpoint-500" \
    "${sys_dir}/checkpoint-1000" \
    "${sys_dir}/checkpoint-1500" \
    "${sys_dir}/checkpoint-2000" \
    "${sys_dir}/checkpoint-2500" \
    "${sys_dir}/checkpoint-3000" \
    "${sys_dir}/checkpoint-3500" \
    "${sys_dir}/checkpoint-4000" \
    "${sys_dir}/checkpoint-4500" \
    "${sys_dir}/checkpoint-5000" \
    "${sys_dir}/checkpoint-5500" \
    "${sys_dir}/checkpoint-6000"
}



# create_checkpoints()(
#     local sys=$1
#     echo "${sys}" \
#     "${sys}/checkpoint-1000" \
#     "${sys}/checkpoint-10000" \
#     "${sys}/checkpoint-10500" \
#     "${sys}/checkpoint-11000" \
#     "${sys}/checkpoint-11500" \
#     "${sys}/checkpoint-12000" \
#     "${sys}/checkpoint-12500" \
#     "${sys}/checkpoint-13000" \
#     "${sys}/checkpoint-1500"  \
#     "${sys}/checkpoint-2000" \
#     "${sys}/checkpoint-2500" \
#     "${sys}/checkpoint-3000" \
#     "${sys}/checkpoint-3500" \
#     "${sys}/checkpoint-4000" \
#     "${sys}/checkpoint-4500" \
#     "${sys}/checkpoint-500" \
#     "${sys}/checkpoint-5000" \
#     "${sys}/checkpoint-5500" \
#     "${sys}/checkpoint-6000" \
#     "${sys}/checkpoint-6500" \
#     "${sys}/checkpoint-7000" \
#     "${sys}/checkpoint-7500" \
#     "${sys}/checkpoint-8000" \
#     "${sys}/checkpoint-8500" \
#     "${sys}/checkpoint-9000" \
#     "${sys}/checkpoint-9500"
# )


checkpoints_nopnx=($(create_checkpoints $sys_nopnx))
checkpoint_nopnx=${checkpoints_nopnx[$SLURM_ARRAY_TASK_ID]}



# python run_m2scorer.py \
#     --system_output $checkpoint_nopnx/dev.ensemble.txt \
#     --m2_file $m2_edits \


# printf "Running M2 Evaluation without pnx on ${checkpoint_nopnx} ...\n"

# python run_m2scorer.py \
#     --system_output $checkpoint_nopnx/dev.ensemble.nopnx.txt \
#     --m2_file $m2_edits_nopnx \


for i in {1..3}
do
    python run_m2scorer.py \
        --system_output $checkpoint_nopnx/dev.txt.${i} \
        --m2_file $m2_edits

    # printf "Running M2 Evaluation without pnx on ${checkpoint_nopnx} ...\n"

    python run_m2scorer.py \
        --system_output $checkpoint_nopnx/dev.txt.${i}.nopnx \
        --m2_file $m2_edits_nopnx

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.nopnx_edit.${i}.pnx_edit.pred_t05.check \
    #     --m2_file $m2_edits \

    # printf "Running M2 Evaluation without pnx on ${checkpoint_nopnx} ...\n"

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.nopnx_edit.${i}.pnx_edit.pred_t05.check.nopnx \
    #     --m2_file $m2_edits_nopnx \

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.nopnx_edit.${i}.nopnx \
    #     --m2_file $m2_edits_nopnx \

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.3.tokens.txt.pnx_edit.${i}.pred_t05 \
    #     --m2_file $m2_edits \

    # python run_m2scorer.py \
    #     --system_output $checkpoint_nopnx/dev.txt.3.tokens.txt.pnx_edit.${i}.pred_t05.nopnx \
    #     --m2_file $m2_edits_nopnx

done