#!/bin/bash -l
#SBATCH --job-name=KD_EffNetV2            # Nom du job
#SBATCH --nodes=1                         # 1 nœud
#SBATCH --ntasks=4                        # 4 tâches (une par GPU)
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4                      # 4 GPUs sur le nœud
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1

# -- IMPORTANT --
# Fixer une seule fois MASTER_PORT pour éviter tout conflit
export MASTER_PORT=41519

# Activation de l'environnement conda
conda activate kiwano_env_resnet34

# Lancement en mode DDP avec srun
# Ajustez le chemin vers kd_train.py en fonction de votre organisation
# Exemple: "utils/kd_train.py" si vous avez copié le script là
# Ajustez également les arguments (teacher_ckpt, student_ckpt, etc.) selon vos besoins

srun --ntasks=4 --gpus=4 --cpus-per-task=10 \
  python3 utils/kd_train.py \
    --teacher_ckpt /chemin/vers/teacher_effnetB2.ckpt \
    --student_ckpt /chemin/vers/student_effnetB0.ckpt \
    --musan data/musan \
    --rirs_noises data/rirs_noises \
    --temperature 4.0 \
    --alpha 0.5 \
    --use_cosine_loss \
    data/voxceleb2 \
    exp/kd_effnetB0
