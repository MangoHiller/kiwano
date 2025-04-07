#!/bin/bash
#SBATCH --nodes=2
##SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --job-name=Kiwano_resnet101
#SBATCH --cpus-per-task=3
#SBATCH --time=72:00:00
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p2
#SBATCH --account=mke@v100
##SBATCH -C v100-32g
#SBATCH --mail-type=ALL            # Notifications par email
#SBATCH --mail-user=  # Remplace par ton email
#SBATCH --output=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/train_resnet101_%j.out   # Fichier de sortie
#SBATCH --error=/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/logs/train_resnet101_%j.err    # Fichier d'erreur

export OMP_NUM_THREADS=3

module purge
module load pytorch-gpu/py3/2.3.0

# Activation de l'environnement conda
source /lustre/fswork/projects/rech/mke/username/miniconda3/bin/activate kiwano_env_resnet34

# Définition des chemins
export WORK_DIR="/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/"
export EXP_DIR="$WORK_DIR/exp/resnet101_5994"
export LOGS_DIR="$WORK_DIR/logs"
export DATA_DIR="$WORK_DIR/data/data/voxceleb2"

export MUSAN_DIR="$WORK_DIR/data/data/musan"
export RIRS_NOISES_DIR="$WORK_DIR/data/data/rirs_noises"

# Création des répertoires si non existants
mkdir -p $EXP_DIR
mkdir -p $LOGS_DIR

# Exécution de l'entraînement avec redirection des logs
#srun python3 utils/train_resnet.py $DATA_DIR $EXP_DIR
# Exécution de l'entraînement avec les chemins MUSAN et RIRS NOISES
srun python3 utils/train_resnet.py --musan $MUSAN_DIR --rirs_noises $RIRS_NOISES_DIR $DATA_DIR $EXP_DIR

#pr reprendre :
#srun python3 utils/train_resnet.py --checkpoint $EXP_DIR/model77.ckpt --musan $MUSAN_DIR --rirs_noises $RIRS_NOISES_DIR $DATA_DIR $EXP_DIR

conda deactivate