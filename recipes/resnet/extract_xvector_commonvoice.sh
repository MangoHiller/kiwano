#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=CV_Extract       # Nom de job plus spécifique
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00           # Augmenter le temps, CV est gros
#SBATCH -A mke@v100               # Vérifie si ce compte est toujours valide/nécessaire
#SBATCH --qos=qos_gpu-t3          # Vérifie si ce QOS est toujours valide/nécessaire
#SBATCH -C v100-32g               # Garder si le modèle nécessite cette mémoire GPU
#SBATCH --output=logs/extract_resnet_commonvoice_%j.out   # Fichier de sortie spécifique CV
#SBATCH --error=logs/extract_resnet_commonvoice_%j.err    # Fichier d'erreur spécifique CV
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

set -e # Arrête en cas d'erreur
set -u # Erreur si variable non définie
set -o pipefail

# --- Configuration ---
# $1 = Répertoire où se trouvent les checkpoints (ex: /path/to/resnet101)
MODEL_DIR="$1"
# $2 = Tag/Époque du modèle à utiliser
EPOCH_TAG="$2"

# Chemin de base vers les données PRÉPARÉES de CommonVoice
CV_DATA_BASE_DIR="/lustre/fswork/projects/rech/mke/username/kiwano/recipes/resnet/data/data/commonvoice"

# Répertoire de sortie pour les x-vectors CommonVoice
OUTPUT_DIR="${MODEL_DIR}/commonvoice_embeddings.${EPOCH_TAG}"

# Créer le répertoire de sortie principal
mkdir -p "$OUTPUT_DIR"
echo "Répertoire de sortie pour les embeddings : ${OUTPUT_DIR}"

# --- Environnement ---
export OMP_NUM_THREADS=10
export CUDA_LAUNCH_BLOCKING=1 # Utile pour le débogage CUDA, peut être enlevé en production
export NCCL_ASYNC_ERROR_HANDLING=1 # Optionnel

module purge
module load pytorch-gpu/py3/2.2.0  # Ou la version requise par Kiwano/PyTorch

source /lustre/fswork/projects/rech/mke/username/miniconda3/bin/activate kiwano_env_resnet34

echo "Environnement chargé."
echo "Modèle utilisé : ${MODEL_DIR}/model${EPOCH_TAG}.ckpt"

# --- Boucle sur les langues ---
echo "Début de l'extraction des x-vectors pour Common Voice..."

# Trouver tous les répertoires de langue dans CV_DATA_BASE_DIR
# (Exclut les fichiers comme comparisons.txt ou les dossiers comme archives s'ils existent là)
for lang_dir in $(find "$CV_DATA_BASE_DIR" -mindepth 1 -maxdepth 1 -type d); do
    lang_code=$(basename "$lang_dir")
    lang_liste_file="${lang_dir}/liste"
    output_pkl_file="${OUTPUT_DIR}/xvector.${lang_code}.pkl" # Nom de fichier de sortie par langue

    echo ""
    echo ">>> Traitement de la langue : $lang_code <<<"

    # Vérifier si le fichier liste existe pour cette langue
    if [ -f "$lang_liste_file" ]; then
        echo "Fichier liste trouvé : $lang_liste_file"
        echo "Fichier de sortie : $output_pkl_file"

        # Exécuter l'extraction pour cette langue
        # Note: on utilise lang_dir comme data_dir pour extract_resnet.py
        srun python3 utils/extract_resnet.py \
            --world_size=1 \
            --rank=0 \
            "$lang_dir" \
            "${MODEL_DIR}/model${EPOCH_TAG}.ckpt" \
            "pkl:${output_pkl_file}"

        echo "Extraction pour $lang_code terminée."
    else
        echo "ATTENTION: Fichier liste '$lang_liste_file' non trouvé. Langue '$lang_code' ignorée."
    fi
done

echo ""
echo "--- Extraction terminée pour toutes les langues trouvées ---"

# Désactivation de l'environnement conda
conda deactivate