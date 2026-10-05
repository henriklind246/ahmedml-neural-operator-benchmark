#!/bin/bash -l
#SBATCH --job-name=ahmedml_full_geometry
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-5
#SBATCH --output=slurm-eval-%A_%a.out
#SBATCH --error=slurm-eval-%A_%a.err

set -euo pipefail

# Submit from the benchmark repository root, after activating the training env.
# REPO_ROOT may override the submission directory. An array task uses one GPU.
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
test -f evaluation/eval_ahmedml.py

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${LRSA_ROOT:-$HOME/repos/LRSA-Operator}/src:${PYTHONPATH:-}"

AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$AHMEDML_ROOT}"
EVAL_ROOT="${EVAL_ROOT:-$AHMEDML_ROOT/evaluation/full_geometry}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-model_500.pth}"

models=(transolver3_full transolver3_efficient linearno_full linearno_efficient lrsa_full lrsa_efficient)
checkpoint_dirs=(t3_full_500 t3_efficient_500 linearno_full_500 linearno_efficient_500 lrsa_full_500 lrsa_efficient_500)
index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "$index" =~ ^[0-5]$ ]]; then
    echo "Expected SLURM_ARRAY_TASK_ID in 0..5" >&2
    exit 2
fi
model="${models[$index]}"
checkpoint="${MODEL_CKPT:-$CHECKPOINT_ROOT/${checkpoint_dirs[$index]}/$CHECKPOINT_NAME}"
test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

echo "Model: $model; checkpoint: $checkpoint; job: ${SLURM_JOB_ID:-local}"
nvidia-smi

# Extra CLI arguments support smoke tests, e.g. --runs run_4.
"${PYTHON_BIN:-python}" -u evaluation/eval_ahmedml.py \
    --model "$model" \
    --data_dir "${DATA_DIR:-$AHMEDML_ROOT/t3_chunks}" \
    --json_file "${SPLIT_FILE:-$PWD/data/splits/ahmedml_test.json}" \
    --norm_stats_file "${NORM_STATS_FILE:-$AHMEDML_ROOT/ahmedml_norm_stats.pkl}" \
    --model_ckpt "$checkpoint" \
    --out_dir "$EVAL_ROOT/$model/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --chunk_size "${CHUNK_SIZE:-50000}" \
    "$@" \
    --inference_mode full_geometry
