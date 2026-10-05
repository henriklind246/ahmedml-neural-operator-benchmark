#!/bin/bash -l
#SBATCH --job-name=ahmedml_full_geometry
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm-eval-%j.out
#SBATCH --error=slurm-eval-%j.err

set -euo pipefail

# Usage: sbatch slurm/eval_full_geometry.slurm MODEL [evaluator arguments]
# Example: sbatch slurm/eval_full_geometry.slurm lrsa_full --runs run_4
if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch slurm/eval_full_geometry.slurm MODEL [evaluator arguments]" >&2
    echo "Models: transolver3_full transolver3_efficient linearno_full linearno_efficient lrsa_full lrsa_efficient" >&2
    exit 2
fi
model="$1"
shift
case "$model" in
    transolver3_full) checkpoint_dir=t3_full_500 ;;
    transolver3_efficient) checkpoint_dir=t3_efficient_500 ;;
    linearno_full) checkpoint_dir=linearno_full_500 ;;
    linearno_efficient) checkpoint_dir=linearno_efficient_500 ;;
    lrsa_full) checkpoint_dir=lrsa_full_500 ;;
    lrsa_efficient) checkpoint_dir=lrsa_efficient_500 ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
esac

# Submit from the benchmark repository root, after activating the training env.
# REPO_ROOT may override the submission directory. Each submission uses one GPU.
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
test -f evaluation/eval_ahmedml.py

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
if [[ "$model" == lrsa_* ]]; then
    export PYTHONPATH="$PWD:${LRSA_ROOT:-$HOME/repos/LRSA-Operator}/src:${PYTHONPATH:-}"
fi

AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$AHMEDML_ROOT}"
EVAL_ROOT="${EVAL_ROOT:-$AHMEDML_ROOT/evaluation/full_geometry}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-model_500.pth}"

checkpoint="${MODEL_CKPT:-$CHECKPOINT_ROOT/$checkpoint_dir/$CHECKPOINT_NAME}"
test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

echo "Model: $model; checkpoint: $checkpoint; job: ${SLURM_JOB_ID:-local}"
nvidia-smi

# Extra CLI arguments support smoke tests, e.g. --runs run_4.
"${PYTHON_BIN:-python}" -u evaluation/eval_ahmedml.py \
    --data_dir "${DATA_DIR:-$AHMEDML_ROOT/t3_chunks}" \
    --json_file "${SPLIT_FILE:-$PWD/data/splits/ahmedml_test.json}" \
    --norm_stats_file "${NORM_STATS_FILE:-$AHMEDML_ROOT/ahmedml_norm_stats.pkl}" \
    --model_ckpt "$checkpoint" \
    --out_dir "$EVAL_ROOT/$model/${SLURM_JOB_ID:-manual}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --chunk_size "${CHUNK_SIZE:-50000}" \
    "$@" \
    --model "$model" \
    --inference_mode full_geometry
