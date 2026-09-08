#!/bin/bash -l

#SBATCH --job-name=t3_eff_ahmedml_500
#SBATCH --partition=msigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=17:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
	
AHMEDML_ROOT="${AHMEDML_ROOT:-/scratch.global/$USER/ahmedml}"

mkdir -p ${AHMEDML_ROOT}/t3_efficient_500

cd ~/repos/Transolver-3

export OMP_NUM_THREADS=4

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  main_ahmedml_surface_efficient.py \
  --data_dir ${AHMEDML_ROOT}/t3_chunks \
  --json_file ./ahmedml_trainval.json \
  --norm_stats_file ${AHMEDML_ROOT}/ahmedml_norm_stats.pkl \
  --save_dir ${AHMEDML_ROOT}/t3_efficient_500 \
  --batch_size 1 \
  --nb_epochs 500 \
  --val_iter 25
