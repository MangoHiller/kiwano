#!/bin/bash -l
#SBATCH --job-name=EffNetV2
#SBATCH --nodes=1                  # 1 nœud (car 2 GPUs suffisent)
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4         # 4 tâches, 1 par GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out        # Fichier de log (standard output)
#SBATCH --error=slurm-%j.err         # Fichier de log (erreurs)


export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

#module purge
#module load pytorch-gpu/py3/2.3.0

# -- IMPORTANT --
# Fixer une seule fois MASTER_PORT, pour que chaque sous-processus voie la même valeur.
export MASTER_PORT=41519


#source ~/.bashrc
conda activate kiwano_env_resnet34

#srun python3 utils/train_resnet.py data/voxceleb2/ exp/resnet/
#srun python3 utils/train_efficientnet.py data/voxceleb2/ exp/efficientnet/

# Lancer le script avec le checkpoint
#srun --ntasks=4 --gpus=4 --cpus-per-task=10 python3 utils/train_efficientnet.py \
#    --checkpoint exp/efficientnet/model_effnet_86.ckpt \
#    --musan data/musan \
#    --rirs_noises data/rirs_noises \
#    data/voxceleb2 \
#    exp/efficientnetB1

srun --ntasks=4 --gpus=4 --cpus-per-task=10 python3 utils/train_efficientnet.py \
    --musan data/musan \
    --rirs_noises data/rirs_noises \
    data/voxceleb2 \
    exp/efficientnetB2
