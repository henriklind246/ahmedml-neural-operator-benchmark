#!/bin/bash
#SBATCH --job-name=linearno500
#SBATCH --mail-user=lind1775
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=21:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Submit from this repository after activating the training environment.
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
test -f training/launchers/linearno_full.py

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"

SAVE_DIR="${SAVE_DIR:-$AHMEDML_ROOT/linearno_full_500}"
mkdir -p "$SAVE_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi

"${PYTHON_BIN:-python}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  training/launchers/linearno_full.py \
  --data_dir "$AHMEDML_ROOT/t3_chunks" \
  --json_file data/splits/ahmedml_trainval.json \
  --norm_stats_file "$AHMEDML_ROOT/ahmedml_norm_stats.pkl" \
  --save_dir "$SAVE_DIR" \
  --nb_epochs 500 \
  --batch_size 1 \
  --lr 0.001 \
  --val_iter 25

echo "Finished: $(date)"
