#!/bin/bash
#SBATCH --job-name=TestLoadEff
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=test_load_eff_%j.out
#SBATCH --error=test_load_eff_%j.err

export OMP_NUM_THREADS=4
set -x

echo "[DEBUG] Début du script test_dataset_0.sh"
source ~/.bashrc
conda activate kiwano_env_resnet34

data_dir=$1        # ex: data/voxceleb1/
model_ckpt=$2      # ex: exp/efficientnet/model_effnet_12.ckpt

python3 utils/test_dataset_0.py \
    --world_size=1 \
    --rank=0 \
    "${data_dir}" \
    "${model_ckpt}"

echo "[DEBUG] Fin du script test_dataset_0.sh"
