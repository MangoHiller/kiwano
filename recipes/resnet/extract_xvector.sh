#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=Kiwano
#SBATCH --cpus-per-task=10
#SBATCH --time=2:20:00
#SBATCH -A mke@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --output=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/extract_resnet_50_PSEUDOL_distill_%j.out   # Fichier de sortie
#SBATCH --error=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/extract_resnet_50_PSEUDOL_distill_%j.err   # Fichier d'erreur

#module load pytorch-gpu/py3/1.7.1+nccl-2.8.3-1

export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

module purge
module load pytorch-gpu/py3/2.2.0

source /lustre/fswork/projects/rech/mke/username/miniconda3/bin/activate kiwano_env_resnet34

################################################################################
# Paramètres
################################################################################
# $1 = répertoire où se trouvent les checkpoints (ex: /path/to/resnet101).
MODEL_DIR="$1"
epoch_tag=$2 

mkdir -p ${MODEL_DIR}/voxceleb1.${epoch_tag}/

# 2) Chemin vers les données (VOXCELEB1)
DATA_DIR="/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/data/data/voxceleb1"



################################################################################
# Exécution du script Python d'extraction
################################################################################

python3 utils/extract_resnet.py --world_size=1 --rank=0 $DATA_DIR ${MODEL_DIR}/model${epoch_tag}.ckpt pkl:${MODEL_DIR}/voxceleb1.${epoch_tag}/xvector.${epoch_tag}.pkl

########################
# ANCIEN SCRIPT :
#
#dir=$1

# Nom du checkpoint à charger (en fonction de la tâche)
#CKPT_FILE="model${SLURM_ARRAY_TASK_ID}.ckpt"

#mkdir -p ${dir}/voxceleb1.${2}/

#python3 utils/extract_resnet.py --world_size=$SLURM_ARRAY_TASK_COUNT  --rank=$SLURM_ARRAY_TASK_ID data/voxceleb1/ ${dir}/model${CKPT_FILE}.ckpt pkl:${dir}/voxceleb1.${CKPT_FILE}/xvector.$SLURM_ARRAY_TASK_ID.pkl
############################