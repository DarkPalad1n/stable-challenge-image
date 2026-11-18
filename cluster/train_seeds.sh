#!/bin/bash
# Train one SwAV configuration with seeds 0-5 in parallel on the Brown CS Hydra cluster, one GPU each.
# Scoring with challenge_score.py runs afterwards in one CPU job (score_models.sh), so no GPU is held
# idle.
#
#   cd ~/stable-challenge-image
#   sbatch cluster/train_seeds.sh NAME [MAIN.PY OPTIONS ...]
#
# NAME labels the configuration; the options are passed to main.py and override the ones set below.
# Outputs: runs/hydra-NAME-s{0..5}/ and slurm-swav-JOBID_SEED.log
#
# The resources below fit main.py's defaults (batch 128, 200 epochs: 2.7-4.7 hours per seed). Change
# them on the command line, before the script name:
#   --gres=gpu:geforce_gtx_2080_ti:1   11 GB, enough for batch 32 but not for batch 128
#   --gres=gpu:nvidia_l40s:1           for batch 256
#   --time=30:00:00                    for 800 epochs
#   --exclude=gpu1903                  to avoid a node (this one once ran three times slower)
#SBATCH --job-name=swav
#SBATCH --partition=gpus
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail

name=$1
shift
seed=$SLURM_ARRAY_TASK_ID
run_dir="runs/hydra-$name-s$seed"

cd "$SLURM_SUBMIT_DIR"
echo "node $(hostname), commit $(git rev-parse --short HEAD), seed $seed, options: $*"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name --format=csv,noheader

# One loader worker per CPU of the job. Evaluating every 5 epochs records the training curve; an
# evaluation does not change the training.
"$HOME/.local/bin/uv" run --locked main.py --data data/train.parquet --output "$run_dir" --seed "$seed" \
    --num-workers 4 --eval-every 5 "$@"
