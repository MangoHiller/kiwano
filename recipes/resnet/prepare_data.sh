#!/bin/bash
#SBATCH --job-name=prepare_data_kiwano   # Nom du job
#SBATCH --output=logs/prepare_commonvoice_%j.out   # Fichier de sortie
#SBATCH --error=logs/prepare_commonvoice_%j.err    # Fichier d'erreur
#SBATCH --partition=prepost    # Partition dédiée aux pré/post-traitements
#SBATCH --account=mke@cpu
#SBATCH --nodes=1              # Réserver un nœud
#SBATCH --ntasks=1             # Une seule tâche
#SBATCH --cpus-per-task=16     # Allouer 16 CPU
#SBATCH --time=20:00:00        # Temps maximum d'exécution (ajustable)
#SBATCH --hint=nomultithread   # Désactiver l'hyperthreading pour de meilleures perfs
#SBATCH --mail-type=ALL        # Notifications par email sur l’état du job
#SBATCH --mail-user=  # Remplace par ton email

# Chargement de l'environnement
module purge
module load ffmpeg/6.1.1       # Charge FFMPEG pour traiter les fichiers audio

# Activation de l'environnement conda
source /lustre/fswork/projects/rech/mke/username/miniconda3/bin/activate kiwano_env_resnet34

# Création des répertoires de logs si nécessaire
mkdir -p logs

# Définition des chemins pour stocker les données dans $SCRATCH
SCRATCH_DATA="/lustre/fsn1/projects/rech/mke/username/kiwano_data"
WORK_DATA="/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/data"

#création des répertoires dans $SCRATCH
mkdir -p $SCRATCH_DATA

# Suppression de l'ancien lien symbolique dans $WORK (si existant)
#if [ -d "$WORK_DATA" ] || [ -L "$WORK_DATA" ]; then
#    rm -rf "$WORK_DATA"
#fi

# Création d'un lien symbolique dans $WORK pointant vers $SCRATCH
#ln -s $SCRATCH_DATA $WORK_DATA

# Exécution des scripts de téléchargement et de préparation
echo "Début du téléchargement et de la préparation des données VoxCeleb1 et VoxCeleb2 et Commonvoice"

### Téléchargement et nettoyage de CommonVoice
echo "Téléchargement de CommonVoice"
srun python local/download_commonvoice21.py $SCRATCH_DATA/db/commonvoice

echo "Préparation de CommonVoice"
srun python local/prepare_commonvoice21.py \
    $SCRATCH_DATA/db/commonvoice \
    $SCRATCH_DATA/db/commonvoice/comparisons.txt \
    $SCRATCH_DATA/data/commonvoice \
    --vad \
    --num_jobs 16 \
    --sampling_frequency 16000 \
    --langs mr gn rw

#srun python local/download_voxceleb1.py $SCRATCH_DATA/db/voxceleb1/
#srun python local/prepare_voxceleb1.py --vad --num_jobs 30 $SCRATCH_DATA/db/voxceleb1/ $SCRATCH_DATA/data/voxceleb1/

#srun python local/download_voxceleb2.py --num_jobs 30 $SCRATCH_DATA/db/voxceleb2/
#srun python local/prepare_voxceleb2.py --vad --num_jobs 30 $SCRATCH_DATA/db/voxceleb2/ $SCRATCH_DATA/data/voxceleb2/

# Optionnel : Préparer MUSAN et RIRS NOISES
#srun python local/download_musan.py $SCRATCH_DATA/db/musan/
#srun python local/prepare_musan.py $SCRATCH_DATA/db/musan/ $SCRATCH_DATA/data/musan/

#srun python local/download_rirs_noises.py $SCRATCH_DATA/db/rirs_noises/
#srun python local/prepare_rirs_noises.py $SCRATCH_DATA/db/rirs_noises/ $SCRATCH_DATA/data/rirs_noises/

echo "Fin de l'exécution du script"

# Désactivation de l'environnement conda après exécution
conda deactivate