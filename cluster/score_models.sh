#!/bin/bash
# Score the models of one configuration (all seeds) with challenge_score.py in one CPU job on Hydra.
#
#   cd ~/stable-challenge-image
#   sbatch cluster/score_models.sh NAME          the final model of every seed
#   sbatch cluster/score_models.sh NAME curve    the models kept by main.py --keep-evaluated-models
#
# Outputs: runs/hydra-challenge-scores-NAME.json or runs/hydra-challenge-curve-NAME.json, and
# slurm-score-JOBID.log with the table. cluster/summarize.py reads both JSON files.
#SBATCH --job-name=score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail

name=$1
if [ "${2:-}" = curve ]; then
    models="model-epoch*.onnx"
    out="runs/hydra-challenge-curve-$name.json"
else
    models="model.onnx"
    out="runs/hydra-challenge-scores-$name.json"
fi

cd "$SLURM_SUBMIT_DIR"
# The pattern is left unquoted on purpose: the shell expands it to the model files.
"$HOME/.local/bin/uv" run --locked challenge_score.py runs/hydra-"$name"-s*/$models \
    --data data/train.parquet --out "$out"
