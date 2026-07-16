#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=24:00:00
#SBATCH --job-name=elf-a1-frozen-t5-l256-s42-h100-v1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export APPTAINER_CACHEDIR=/datapool/researcher/elf/apptainer/cache
export APPTAINER_TMPDIR=/datapool/researcher/elf/apptainer/tmp
export BACKEND_JOB_ID="$SLURM_JOB_ID"
mkdir -p /datapool/researcher/elf/runs/elf-a1-frozen-t5-l256-s42-h100-v1
attempt_log_dir=/datapool/researcher/elf/runs/elf-a1-frozen-t5-l256-s42-h100-v1/attempts/attempt-001
mkdir -p "$attempt_log_dir"
exec > >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.out")      2> >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.err" >&2)
test -d /datapool/researcher/elf/sources/e44db5b272df3b6a196e8f58addc54446003595f
test -s /datapool/researcher/elf/images/1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4.sif
srun apptainer exec --nv   --bind /datapool:/datapool   --bind /datapool/researcher/elf/sources/e44db5b272df3b6a196e8f58addc54446003595f:/app   --pwd /app   /datapool/researcher/elf/images/1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4.sif   env ATTEMPT_ID=attempt-001 BACKEND=slurm CHECKPOINT_ROOT=/datapool/researcher/elf/checkpoints DATA_ROOT=/datapool/researcher ELF_B_OWT_CHECKPOINT=/datapool/researcher/elf/checkpoints/ELF-B-owt-torch/checkpoint_95085 HF_DATASETS_CACHE=/datapool/researcher/.cache/huggingface/datasets HF_HOME=/datapool/researcher/.cache/huggingface IMAGE_ID=sha256:1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4 LOG_FREQ=100 NGPU=4 NUM_WORKERS=8 OUTPUT_DIR=/datapool/researcher/elf/runs/elf-a1-frozen-t5-l256-s42-h100-v1 PROJECT_DATA_ROOT=/datapool/researcher/elf PROJECT_NAME=elf QUOTA_TYPE=normal REQUIRE_OFFLINE_CACHE=1 RUN_ID=elf-a1-frozen-t5-l256-s42-h100-v1 SAVE_DIR=/datapool/researcher/elf/saved_models SOURCE_ID=e44db5b272df3b6a196e8f58addc54446003595f USE_WANDB=false WANDB_CACHE_DIR=/datapool/researcher/elf/wandb_cache WANDB_DIR=/datapool/researcher/elf/wandb bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_1_sentence_t5_len256.yml --config_override epochs=1 --config_override grad_accum_steps=1 --config_override save_freq=0.1 --config_override eval_freq=1 --config_override num_samples=256 --config_override reconstruction_num_samples=256 --config_override train_sampling_eval_freq=2000 --config_override train_sampling_eval_num_samples=64 --config_override train_sampling_eval_batch_size=16
