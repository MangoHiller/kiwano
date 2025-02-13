#!/bin/bash -l
#SBATCH --job-name=DDP
#SBATCH --partition=gpu
#SBATCH --nodes=2                  # Demande 2 nœuds
#SBATCH --ntasks-per-node=1        # 1 tâche par nœud
#SBATCH --gres=gpu:1               # 1 GPU par nœud
#SBATCH --cpus-per-task=4          # 4 CPUs par tâche
#SBATCH --mem=16G                  # 16 Go de RAM
#SBATCH --time=00:10:00            # Temps maximum : 10 minutes
#SBATCH --output=hostlist_test_%j.out
#SBATCH --error=hostlist_test_%j.err


export OMP_NUM_THREADS=4
export MASTER_PORT=29500

conda activate kiwano_env_resnet34  # Active l'environnement conda

# Exécute le script Python
srun python3 ddp_test.py
