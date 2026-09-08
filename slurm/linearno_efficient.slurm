#!/bin/bash
#SBATCH --job-name=lnoeff500
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

cd ~/repos/Transolver-3

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=8

AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"

mkdir -p ${AHMEDML_ROOT}/linearno_efficient_500

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  main_ahmedml_linearno_efficient.py \
  --data_dir ${AHMEDML_ROOT}/t3_chunks \
  --json_file ./ahmedml_trainval.json \
  --norm_stats_file ${AHMEDML_ROOT}/ahmedml_norm_stats.pkl \
  --save_dir ${AHMEDML_ROOT}/linearno_efficient_500 \
  --nb_epochs 500 \
  --batch_size 1 \
  --lr 0.001 \
  --val_iter 25

echo "Finished: $(date)"
