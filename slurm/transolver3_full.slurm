#!/bin/bash -l

#SBATCH --job-name=t3_ahmedml_500
#SBATCH --mail-user=lind1775
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
	
AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"

mkdir -p ${AHMEDML_ROOT}/t3_full_500

cd ~/repos/Transolver-3

export OMP_NUM_THREADS=4

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  main_ahmedml_surface.py \
  --data_dir ${AHMEDML_ROOT}/t3_chunks \
  --json_file ./ahmedml_trainval.json \
  --norm_stats_file ${AHMEDML_ROOT}/ahmedml_norm_stats.pkl \
  --save_dir ${AHMEDML_ROOT}/t3_full_500 \
  --batch_size 1 \
  --nb_epochs 500 \
  --val_iter 25
