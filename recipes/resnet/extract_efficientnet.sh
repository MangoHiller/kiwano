#!/bin/bash
#SBATCH --job-name=EffNet_extract
#SBATCH --partition=gpu
#SBATCH --nodelist=idyie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=extract_effnet_%j.out
#SBATCH --error=extract_effnet_%j.err

# Ce script lance l'extraction des embeddings (xvectors) pour VoxCeleb1
# depuis un modèle EfficientNetV2 donné (checkpoint .ckpt).
# Il s'appuie sur un script Python "extract_efficientnet.py".

export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

set -x
echo "[DEBUG] Début du script extract_efficientnet-B2.sh"
#module purge
#module load pytorch-gpu/py3/2.2.0
source ~/.bashrc
conda activate kiwano_env_resnet34

dir=$1          # par ex: exp/efficientnet
epoch_tag=$2    # par ex: 12  => model_effnet_12.ckpt

mkdir -p ${dir}/voxceleb1.${epoch_tag}/

# On peut lancer en mode array si on veut un data split sur N GPU/CPU
# Ici, on lance 1 job => rank=0, world_size=1
#sbatch extract_efficientnet.sh exp/efficientnetB1/ 3
python3 utils/extract_efficientnet.py \
    --world_size=1 \
    --rank=0 \
    data/voxceleb1/ \
    ${dir}/model_effnet_${epoch_tag}.ckpt \
    pkl:${dir}/voxceleb1.${epoch_tag}/xvector.0.pkl
