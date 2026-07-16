#!/usr/bin/env bash
#SBATCH --partition=l40s
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:10:00
#SBATCH --job-name=elf-smoke-slurm-l40s-probe-20260712T0105--attempt-001
#SBATCH --comment=backend-smoke-slurm-probe-20260712T0105/elf-smoke-slurm-l40s-probe-20260712T0105/attempt-001
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export APPTAINER_CACHEDIR=/data/researcher/elf/apptainer/cache
export APPTAINER_TMPDIR=/data/researcher/elf/apptainer/tmp
export BACKEND_JOB_ID="$SLURM_JOB_ID"
mkdir -p /data/researcher/elf/runs/elf-smoke-slurm-l40s-probe-20260712T0105
attempt_log_dir=/data/researcher/elf/runs/elf-smoke-slurm-l40s-probe-20260712T0105/attempts/attempt-001
mkdir -p "$attempt_log_dir"
exec > >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.out") \
     2> >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.err" >&2)
test -d /data/researcher/elf/sources/runtime.e0aea8892dc87d62a89fc411194599c6e8bcb663f65a74096c3b3b555012d1b0
test -s /data/researcher/elf/images/1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4.sif
srun apptainer exec --nv \
  --bind /data:/data \
  --bind /data/researcher/elf/sources/runtime.e0aea8892dc87d62a89fc411194599c6e8bcb663f65a74096c3b3b555012d1b0:/app \
  --pwd /app \
  /data/researcher/elf/images/1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4.sif \
  env ATTEMPT_ID=attempt-001 BACKEND=slurm BATCH_SIZE=4 CAMPAIGN_ID=campaign.9e9caaaa231bcb4a5a7389988b2dc7bb486219401792b7b14105d7a82e3f9de7 CAMPAIGN_NAME=backend-smoke-slurm-probe-20260712T0105 CHECKPOINT_ROOT=/data/researcher/elf/checkpoints DATA_ROOT=/data/researcher ELF_B_OWT_CHECKPOINT=/data/researcher/elf/checkpoints/ELF-B-owt-torch/checkpoint_95085 GIT_COMMIT=8a525ae08b349cab6b69bf39af48f5e51850521a HF_DATASETS_CACHE=/data/researcher/elf/cache/huggingface/datasets HF_HOME=/data/researcher/elf/cache/huggingface IMAGE_ID=sha256:1e25f456cd26219ff89015a41b8653efb0db4444f911522802f850a791710de4 LOG_FREQ=10 MAX_INFRA_RETRIES=0 NGPU=1 NUM_WORKERS=0 OUTPUT_DIR=/data/researcher/elf/runs/elf-smoke-slurm-l40s-probe-20260712T0105 PROJECT_DATA_ROOT=/data/researcher/elf PROJECT_NAME=elf QUOTA_TYPE=normal REQUIRE_OFFLINE_CACHE=1 RUNTIME_TREE_ID=runtime.e0aea8892dc87d62a89fc411194599c6e8bcb663f65a74096c3b3b555012d1b0 RUN_ID=elf-smoke-slurm-l40s-probe-20260712T0105 SAVE_DIR=/data/researcher/elf/saved_models SOURCE_ID=runtime.e0aea8892dc87d62a89fc411194599c6e8bcb663f65a74096c3b3b555012d1b0 USE_WANDB=false WANDB_CACHE_DIR=/data/researcher/elf/wandb_cache WANDB_DIR=/data/researcher/elf/wandb bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml --config_override data_path=/data/researcher/elf/datasets/smoke-owt-tokenized-v1 --config_override epochs=1 --config_override online_eval=false --config_override reconstruction_eval=false --config_override num_samples=4
