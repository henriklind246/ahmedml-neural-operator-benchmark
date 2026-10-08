#!/bin/bash
#SBATCH --job-name=lrsa500
#SBATCH --mail-user=lind1775
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

cd ~/repos/Transolver-3

# Make official LRSA implementation importable.
export PYTHONPATH="$HOME/repos/LRSA-Operator/src:$PYTHONPATH"

# 8 CPU threads per GPU.
export OMP_NUM_THREADS=8

echo "========================================"
echo "LRSA AhmedML 500-epoch training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

nvidia-smi

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  main_ahmedml_lrsa.py \
  --data_dir ${AHMEDML_ROOT}/t3_chunks \
  --json_file ./ahmedml_trainval.json \
  --norm_stats_file ${AHMEDML_ROOT}/ahmedml_norm_stats.pkl \
  --save_dir ${AHMEDML_ROOT}/lrsa_full_500 \
  --nb_epochs 500 \
  --batch_size 1 \
  --lr 0.001 \
  --val_iter 25

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
