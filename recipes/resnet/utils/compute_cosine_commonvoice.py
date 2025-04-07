import argparse
from kiwano.utils import read_keys
import torch
from kiwano.embedding import read_pkl

import torch # Assure-toi que torch est importé

def scoring_xvector(keys, xvectors_enrollment, xvectors_test):
    """
    Calcule et affiche la similarité cosinus pour les paires d'essais.
    Gère les clés au format '{lang}@clips@{segment_id}' trouvées dans comparisons.txt.

    Args:
        keys: Dictionnaire lu depuis le fichier d'essais (ex: comparisons.txt).
              Clé = tuple (nom_audio1, nom_audio2) avec préfixe.
              Valeur = label (0 ou 1).
        xvectors_enrollment: Dictionnaire d'embeddings {segment_id: tensor}.
                             Peut être le même que xvectors_test.
        xvectors_test: Dictionnaire d'embeddings {segment_id: tensor}.
    """
    cos = torch.nn.CosineSimilarity(dim=0)
    missing_enroll = 0
    missing_test = 0
    processed_pairs = 0

    print(f"Début du scoring pour {len(keys)} paires...")

    for names in keys:
        enrollmentNameWithPrefix = names[0]
        testNameWithPrefix = names[1]

        # Extraire le segment ID en enlevant le préfixe '{lang}@clips@'
        try:
            enrollmentKey = enrollmentNameWithPrefix.split('@clips@')[1]
        except IndexError:
            print(f"ERREUR: Format de clé inattendu pour enrollment: {enrollmentNameWithPrefix}")
            continue # Ignore cette paire

        try:
            testKey = testNameWithPrefix.split('@clips@')[1]
        except IndexError:
            print(f"ERREUR: Format de clé inattendu pour test: {testNameWithPrefix}")
            continue # Ignore cette paire

        # Récupérer les x-vectors en utilisant les clés nettoyées
        xvectorEnrollment = xvectors_enrollment.get(enrollmentKey)
        xvectorTest = xvectors_test.get(testKey)

        # Vérifier si les x-vectors ont été trouvés
        if xvectorEnrollment is None:
            # print(f"ATTENTION: X-vector non trouvé pour enrollment: {enrollmentKey} (venant de {enrollmentNameWithPrefix})")
            missing_enroll += 1
            continue # Ignore cette paire si un vecteur manque
        if xvectorTest is None:
            # print(f"ATTENTION: X-vector non trouvé pour test: {testKey} (venant de {testNameWithPrefix})")
            missing_test += 1
            continue # Ignore cette paire

        # Calculer et afficher le score
        try:
             # S'assurer que les tenseurs sont sur CPU si besoin, et sont des float32
             if not isinstance(xvectorEnrollment, torch.Tensor):
                 xvectorEnrollment = torch.tensor(xvectorEnrollment, dtype=torch.float32)
             if not isinstance(xvectorTest, torch.Tensor):
                 xvectorTest = torch.tensor(xvectorTest, dtype=torch.float32)

             score = cos(xvectorEnrollment.cpu(), xvectorTest.cpu())
             # Imprimer avec les noms ORIGINAUX du fichier d'essai
             print(f"{enrollmentNameWithPrefix} {testNameWithPrefix} {score.item()}")
             processed_pairs += 1
        except Exception as e:
             print(f"ERREUR lors du calcul du score pour la paire ({enrollmentNameWithPrefix}, {testNameWithPrefix}): {e}")


    print(f"\nScoring terminé.")
    print(f"Paires traitées avec succès : {processed_pairs}")
    if missing_enroll > 0:
        print(f"ATTENTION: {missing_enroll} x-vectors d'enrollment manquants.")
    if missing_test > 0:
        print(f"ATTENTION: {missing_test} x-vectors de test manquants.")
    if missing_enroll > 0 or missing_test > 0:
         print("=> Cela peut arriver si certains segments de comparisons.txt n'ont pas pu être extraits/préparés.")





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('keys', metavar='keys', type=str,
                        help='the path to the file where the keys are stocked')
    parser.add_argument('xvectorEnrollment', metavar='xvectorEnrollment', type=str,
                        help='command to gather xvectors enrollment in pkl format')
    parser.add_argument('xvectorTest', metavar='xvectorTest', type=str,
                        help='command to gather xvectors test in pkl format')

    args = parser.parse_args()
    trials = read_keys(args.keys)
    enrollment = read_pkl(args.xvectorEnrollment)
    test = read_pkl(args.xvectorTest)

    scoring_xvector(trials, enrollment, test)


