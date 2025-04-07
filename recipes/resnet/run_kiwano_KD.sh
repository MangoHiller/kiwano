#!/bin/bash
###############################################################################
# Exemple de script Slurm pour lancer une distillation (pseudo-label ou MSE)
# sur un ResNet "Student" à partir d'un "Teacher" checkpointé.
###############################################################################
##SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --job-name=distillation_resnet_Feature_MSE
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --partition=gpu_p13
#SBATCH --account=mke@v100
#SBATCH -C v100-32g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --output=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/distill_resnet_Feature_MSE_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/distill_resnet_Feature_MSE_%j.err

###############################################################################
# Nombre de threads OpenMP
###############################################################################
export OMP_NUM_THREADS=10

###############################################################################
# Chargement des modules
###############################################################################
module purge
module load pytorch-gpu/py3/2.3.0

###############################################################################
# Activation de l'environnement conda
###############################################################################
source /lustre/fswork/projects/rech/mke/username/miniconda3/bin/activate kiwano_env_resnet34

###############################################################################
# Définition des chemins
###############################################################################
export WORK_DIR="/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet"
export EXP_DIR="$WORK_DIR/exp/distillation_resnet_feature_mse"         # nouveau répertoire d'output
export LOGS_DIR="$WORK_DIR/logs"
export DATA_DIR="$WORK_DIR/data/data/voxceleb2"
export MUSAN_DIR="$WORK_DIR/data/data/musan"
export RIRS_NOISES_DIR="$WORK_DIR/data/data/rirs_noises"

mkdir -p "$EXP_DIR"
#mkdir -p "$LOGS_DIR"

###############################################################################
# Lancement de l'entraînement avec distillation
# --teacher_checkpoint : chemin du Teacher (ex. ResNet101)
# --kd_mode : 'mse' ou 'pseudo_label' ou 'mse_emb' ou 'cos_emb'ou 'hinton_kd'
###############################################################################
srun python -u utils/train_resnet_KD.py \
  --musan "$MUSAN_DIR" \
  --rirs_noises "$RIRS_NOISES_DIR" \
  --teacher_checkpoint "/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/exp/resnet101_5994/model48.ckpt" \
  --kd_mode "feature_mse" \
  --alpha 0.5 \
  "$DATA_DIR" \
  "$EXP_DIR"

###############################################################################
# Désactivation de l'environnement conda
###############################################################################
conda deactivate
