#!/bin/bash

# $1 = Répertoire où se trouvent les checkpoints ET les embeddings extraits (ex: exp/resnet_v2h_perturb_speed/)
MODEL_DIR="$1"
# $2 = Tag/Époque du modèle utilisé pour l'extraction
EPOCH_TAG="$2"

# Chemin de base vers les données BRUTES de CommonVoice (pour trouver comparisons.txt)
CV_DB_DIR="/lustre/fswork/projects/rech/mke/username/kiwano_data/db/commonvoice"

# Chemin vers le répertoire contenant les embeddings PKL extraits pour CommonVoice
CV_EMBEDDING_DIR="${MODEL_DIR}/commonvoice_embeddings.${EPOCH_TAG}"

# Fichier de comparaison Common Voice
CV_TRIALS_FILE="${CV_DB_DIR}/comparisons.txt"

# Fichier de sortie pour les scores Common Voice
SCORE_FILE="${CV_EMBEDDING_DIR}/scores_commonvoice.${EPOCH_TAG}.txt"

set -e
set -u
set -o pipefail

echo "--- Calcul des scores pour Common Voice ---"
echo "Répertoire des embeddings : ${CV_EMBEDDING_DIR}"
echo "Fichier d'essais : ${CV_TRIALS_FILE}"
echo "Fichier de scores : ${SCORE_FILE}"

# Vérifier si le répertoire d'embeddings existe
if [ ! -d "$CV_EMBEDDING_DIR" ]; then
    echo "ERREUR: Répertoire d'embeddings ${CV_EMBEDDING_DIR} non trouvé."
    exit 1
fi

# Vérifier si le fichier d'essais existe
if [ ! -f "$CV_TRIALS_FILE" ]; then
    echo "ERREUR: Fichier d'essais ${CV_TRIALS_FILE} non trouvé."
    exit 1
fi

# Combiner tous les fichiers pkl de langue en un seul flux pour compute_cosine.py
# L'option "p:..." indique à kiwano de lire depuis un pipe
COMBINED_PKL_CMD="pkl:cat ${CV_EMBEDDING_DIR}/xvector.*.pkl"

# Calculer les scores cosinus
# On passe le même flux combiné pour enrollment et test car comparisons.txt
# peut référencer n'importe quel fichier comme enrollment ou test.
python3 utils/compute_cosine_commonvoice.py \
    "$CV_TRIALS_FILE" \
    "$COMBINED_PKL_CMD" \
    "$COMBINED_PKL_CMD" > "$SCORE_FILE"

echo "Scores calculés dans ${SCORE_FILE}"

# Calculer et afficher EER, DCF, Cllr
echo "--- Évaluation Common Voice ---"
echo "EER:"
python3 utils/compute_eer.py "$CV_TRIALS_FILE" "$SCORE_FILE"
echo "minDCF (p_target=0.01):"
python3 utils/compute_dcf.py --p-target 0.01 "$CV_TRIALS_FILE" "$SCORE_FILE"
echo "Cllr:"
python3 utils/compute_cllr.py "$CV_TRIALS_FILE" "$SCORE_FILE"

echo "--- Fin du scoring pour Common Voice ---"