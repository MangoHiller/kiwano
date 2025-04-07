#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
# from kiwano.utils import Pathlike # Pas utilisé directement ici, mais peut-être par d'autres imports?
import torch
import numpy as np
from subprocess import PIPE, run
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
import os # Ajout nécessaire

# Gestion des imports conditionnels si silero_vad n'est pas toujours nécessaire
try:
    from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks
    _SILERO_AVAILABLE = True
except ImportError:
    _SILERO_AVAILABLE = False
    print("Avertissement: Silero VAD non trouvé. L'option --vad ne fonctionnera pas.")

# Charger le modèle Silero VAD une seule fois SI VAD demandé et disponible
model_vad = None # Initialiser à None

# Dictionnaire des langues - peut rester pour référence mais moins critique maintenant
LANG_CODE = {
    'ab': 'ab', 'ar': 'ar', 'as': 'as', 'ba': 'ba', 'bas': 'bas', 'be': 'be', 'bg': 'bg', 'bn': 'bn',
    'br': 'br', 'ckb': 'ckb', 'cnh': 'cnh', 'cs': 'cs', 'cv': 'cv', 'cy': 'cy', 'da': 'da', 'de': 'de',
    'dv': 'dv', 'el': 'el', 'en': 'en', 'eo': 'eo', 'es': 'es', 'et': 'et', 'eu': 'eu', 'fa': 'fa',
    'fi': 'fi', 'fr': 'fr', 'fy-NL': 'fy-NL', 'ga-IE': 'ga-IE', 'gl': 'gl', 'gn': 'gn', 'ha': 'ha',
    'he': 'he', 'hi': 'hi', 'hu': 'hu', 'hy-AM': 'hy-AM', 'ia': 'ia', 'id': 'id', 'it': 'it', 'ja': 'ja',
    'ka': 'ka', 'kab': 'kab', 'kk': 'kk', 'kmr': 'kmr', 'ky': 'ky', 'lg': 'lg', 'lij': 'lij', 'lt': 'lt',
    'ltg': 'ltg', 'lv': 'lv', 'mhr': 'mhr', 'ml': 'ml', 'mn': 'mn', 'mr': 'mr', 'mrj': 'mrj', 'mt': 'mt',
    'myv': 'myv', 'nan-tw': 'nan-tw', 'ne-NP': 'ne-NP', 'nl': 'nl', 'nn-NO': 'nn-NO', 'oc': 'oc',
    'or': 'or', 'pa-in': 'pa-in', 'pl': 'pl', 'ps': 'ps', 'pt': 'pt', 'rm-sursilv': 'rm-sursilv',
    'rm-vallader': 'rm-vallader', 'ro': 'ro', 'ru': 'ru', 'rw': 'rw', 'sah': 'sah', 'sat': 'sat',
    'sk': 'sk', 'skr': 'skr', 'sl': 'sl', 'sq': 'sq', 'sr': 'sr', 'sv-SE': 'sv-SE', 'sw': 'sw',
    'ta': 'ta', 'th': 'th', 'tk': 'tk', 'tok': 'tok', 'tr': 'tr', 'tt': 'tt', 'ug': 'ug', 'uk': 'uk',
    'ur': 'ur', 'uz': 'uz', 'vi': 'vi', 'yo': 'yo', 'yue': 'yue', 'zh-CN': 'zh-CN',
    'zh-HK': 'zh-HK', 'zh-TW': 'zh-TW'
}

def ffmpeg(file_path: Path, sampling_frequency: int):
    """Converts audio using ffmpeg and returns a Torch tensor."""
    cmd = f"ffmpeg -y -threads 1 -i '{file_path}' -acodec pcm_s16le -ac 1 -ar {sampling_frequency} -f wav -"
    try:
        proc = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, check=True) # Use check=True to raise error if ffmpeg fails
        if proc.stderr:
             # Log ffmpeg errors/warnings non-fatals
             print(f"FFmpeg stderr for {file_path}:\n{proc.stderr.decode()}")
        audio_array = np.frombuffer(proc.stdout, dtype=np.int16)
        if audio_array.size == 0:
             print(f"AVERTISSEMENT: ffmpeg a produit un fichier vide pour {file_path}")
             return None # Retourner None si le fichier est vide
        wav = torch.tensor(audio_array.astype(np.float32) / 32768.0)
        return wav
    except Exception as e:
        print(f"ERREUR lors de l'exécution de ffmpeg pour {file_path}: {e}")
        if hasattr(e, 'stderr'):
            print(f"FFmpeg stderr: {e.stderr.decode()}")
        return None # Retourner None en cas d'erreur

def process_audio(segment_path: Path, client_id: str, out_dir: Path, sampling_frequency: int, vad: bool):
    """Processes a single audio file: ffmpeg conversion, optional VAD, save."""
    global model_vad # Access the globally loaded model
    
    output_spk_dir = out_dir / client_id
    output_spk_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_spk_dir / (segment_path.stem + '.wav')

    # Check if output already exists (optional, for re-running partially)
    # if output_path.exists():
    #     try:
    #         info = sf.info(str(output_path))
    #         duration = round(info.frames / info.samplerate, 2)
    #         return segment_path.stem, client_id, duration, output_path
    #     except Exception:
    #         print(f"Fichier existant {output_path} semble corrompu, re-traitement.")

    wav = ffmpeg(segment_path, sampling_frequency)
    
    # Vérifier si ffmpeg a retourné None (erreur ou fichier vide)
    if wav is None or wav.shape[0] == 0:
        print(f"AVERTISSEMENT: Conversion ffmpeg échouée ou fichier vide pour {segment_path.name}. Ignoré.")
        return None # Important de retourner None pour que la boucle suivante le saute

    if vad and _SILERO_AVAILABLE:
        if model_vad is None: # Charger le modèle VAD seulement si nécessaire et pas déjà chargé
             print("Chargement du modèle Silero VAD...")
             model_vad = load_silero_vad()
        try:
            speech_timestamps = get_speech_timestamps(wav, model_vad, sampling_rate=sampling_frequency, threshold=0.6)
            if speech_timestamps:
                wav = collect_chunks(speech_timestamps, wav)
            #else:
            #    print(f"VAD n'a rien détecté pour {segment_path.name}") # Optionnel: log si VAD vide
        except Exception as e:
            print(f"ERREUR pendant VAD pour {segment_path.name}: {e}")
            # Décider si on garde l'audio original ou si on l'ignore
            # return None # Décommenter pour ignorer si VAD échoue

    # Vérifier à nouveau si le wav est vide après VAD
    if wav.shape[0] == 0:
         print(f"AVERTISSEMENT: Audio vide après VAD pour {segment_path.name}. Ignoré.")
         return None

    try:
        sf.write(output_path, wav.numpy(), sampling_frequency)
        duration = round(wav.shape[0] / sampling_frequency, 2)
        return segment_path.stem, client_id, duration, output_path
    except Exception as e:
        print(f"ERREUR lors de l'écriture WAV pour {output_path}: {e}")
        return None

def clean_and_prepare_commonvoice(cv_root: Path, comparisons_file: Path, language: str, output_dir: Path, sampling_frequency: int, vad: bool, num_jobs: int):
    global model_vad # Pour pouvoir charger le modèle VAD si nécessaire

    lang_folder = cv_root / language # Utilise la casse correcte
    clips_folder = lang_folder / "clips"
    validated_tsv = lang_folder / "validated.tsv"
    output_lang_dir = output_dir / language # Utilise la casse correcte
    output_lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Nettoyage et Préparation pour : {language} ---")

    # 1. Filtrage basé sur comparisons.txt
    print(f"Chargement de comparisons.txt pour filtrer {language}...")
    try:
        # Charger uniquement les colonnes nécessaires pour le filtrage
        comparisons_df = pd.read_csv(comparisons_file, usecols=['audio1', 'audio2'])
    except FileNotFoundError:
         print(f"ERREUR: Fichier comparisons.txt introuvable : {comparisons_file}")
         return
    except Exception as e:
         print(f"ERREUR inattendue lors de la lecture de comparisons.txt: {e}")
         return

    prefix = f"{language}@clips@"
    print(f"Utilisation du préfixe: {prefix}")

    valid_files_from_comparisons = set()
    try:
        if 'audio1' in comparisons_df.columns:
            mask1 = comparisons_df['audio1'].notna() & comparisons_df['audio1'].str.startswith(prefix)
            valid_files_from_comparisons.update(comparisons_df.loc[mask1, 'audio1'].str.replace(prefix, '', regex=False))
        if 'audio2' in comparisons_df.columns:
            mask2 = comparisons_df['audio2'].notna() & comparisons_df['audio2'].str.startswith(prefix)
            valid_files_from_comparisons.update(comparisons_df.loc[mask2, 'audio2'].str.replace(prefix, '', regex=False))
    except KeyError as e:
         print(f"AVERTISSEMENT: Colonne {e} non trouvée dans comparisons.txt")
    except Exception as e:
         print(f"ERREUR inattendue lors du filtrage avec comparisons.txt pour {language}: {e}")
         return

    print(f"{len(valid_files_from_comparisons)} fichiers uniques trouvés dans comparisons.txt pour {language}")
    if not valid_files_from_comparisons:
         print(f"ATTENTION: Aucun fichier trouvé pour le préfixe {prefix} dans comparisons.txt. La langue sera ignorée.")
         return

    # 2. Lister les fichiers MP3 présents
    if not clips_folder.exists():
        print(f"ERREUR: Le dossier clips n'existe pas: {clips_folder}")
        return
    all_files_paths = list(clips_folder.glob('*.mp3'))
    all_files_names = set(p.name for p in all_files_paths)
    print(f"{len(all_files_names)} fichiers MP3 trouvés dans {clips_folder}")

    # 3. Identifier les fichiers à supprimer et ceux à garder
    files_to_keep_names = all_files_names.intersection(valid_files_from_comparisons)
    files_to_delete_paths = [p for p in all_files_paths if p.name not in files_to_keep_names]

    print(f"{len(files_to_keep_names)} fichiers valides à traiter (présents ET listés dans comparisons).")
    print(f"{len(files_to_delete_paths)} fichiers MP3 à supprimer (non listés dans comparisons ou non présents sur disque mais dans la liste initiale).")

    # 4. Suppression des fichiers MP3 inutiles
    if not files_to_delete_paths:
         print("Aucun fichier MP3 invalide à supprimer.")
    else:
         for p in tqdm(files_to_delete_paths, desc=f"Suppression MP3 invalides {language}"):
             try:
                 p.unlink()
             except OSError as e:
                 print(f"Erreur lors de la suppression de {p}: {e}")

    # 5. Charger validated.tsv pour les client_id des fichiers restants
    print(f"Chargement des informations des locuteurs pour {language} depuis {validated_tsv}...")
    if not validated_tsv.exists():
         print(f"ERREUR: Le fichier validated.tsv n'existe pas: {validated_tsv}")
         return
    try:
        # Utilisation de engine='python' pour gérer les erreurs de tokenization
        validated_df = pd.read_csv(
            validated_tsv,
            sep='\t',
            usecols=['path', 'client_id'],
            engine='python', # Correction clé pour mr, gn, rw
            on_bad_lines='warn', # Log les lignes problématiques au lieu de planter
            #low_memory=False
        )
        # Filtrer les lignes où 'path' est NaN (peut arriver avec on_bad_lines='warn')
        validated_df = validated_df.dropna(subset=['path'])
        validated_df = validated_df.set_index('path')
        client_id_dict = validated_df['client_id'].to_dict()
        print(f"Chargement de {len(client_id_dict)} entrées valides depuis {validated_tsv}.")
    except Exception as e:
         print(f"ERREUR critique lors de la lecture de {validated_tsv}: {e}")
         return

    # 6. Préparer la liste des tâches de conversion
    tasks = []
    # Recalculer les chemins valides APRÈS suppression
    valid_paths_after_cleanup = list(clips_folder.glob('*.mp3'))
    skipped_no_id = 0
    for segment_path in valid_paths_after_cleanup:
        # Vérifier si le fichier est bien dans la liste des fichiers à garder
        # (double sécurité, normalement inutile si la suppression a bien marché)
        if segment_path.name in files_to_keep_names:
            client_id = client_id_dict.get(segment_path.name)
            if client_id:
                tasks.append((segment_path, str(client_id), output_lang_dir, sampling_frequency, vad))
            else:
                # Ce warning ne devrait plus apparaître souvent
                # print(f"⚠️ client_id introuvable pour {segment_path.name} dans validated.tsv. Ignoré.")
                skipped_no_id += 1
        # else: # Normalement, ce cas ne devrait plus arriver si la suppression a fonctionné
            # print(f"DEBUG: Fichier {segment_path.name} trouvé sur disque mais pas dans files_to_keep_names?")

    if skipped_no_id > 0:
        print(f"AVERTISSEMENT: {skipped_no_id} fichiers valides ont été ignorés car leur 'path' n'a pas été trouvé dans {validated_tsv}.")
    print(f"Conversion audio WAV ({language}), total {len(tasks)} tâches à lancer.")

    # 7. Lancer le traitement parallèle et écrire le fichier liste
    results = []
    if tasks: # Seulement si on a des tâches à exécuter
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            # Utiliser tqdm sur executor.map pour suivre la progression globale
            futures = [executor.submit(process_audio, *task) for task in tasks]
            for future in tqdm(futures, total=len(tasks), desc=f"Préparation audio {language}"):
                 try:
                     result = future.result()
                     if result: # Vérifier si process_audio n'a pas retourné None
                         results.append(result)
                 except Exception as e:
                      # Log l'erreur mais continue avec les autres si possible
                      print(f"ERREUR dans un worker pour {language}: {e}")

    # 8. Écrire le fichier liste final
    liste_path = output_lang_dir / "liste"
    print(f"Écriture du fichier liste : {liste_path} ({len(results)} entrées)")
    written_count = 0
    with open(liste_path, "w", encoding='utf-8') as liste_file:
         for result in results:
             # result contient (name, spkid, duration, output_path)
             name, spkid, duration, output = result
             # Assurer que le chemin est relatif au dossier data/commonvoice
             try:
                 relative_output_path = Path(output).relative_to(output_dir.parent) # output_dir.parent = data/
                 liste_file.write(f"{name} {spkid} {duration} {relative_output_path}\n")
                 written_count += 1
             except ValueError:
                 # Si le chemin n'est pas relatif (ne devrait pas arriver ici)
                 print(f"ERREUR: Impossible de rendre le chemin relatif: {output}")
                 liste_file.write(f"# ERREUR_PATH: {name} {spkid} {duration} {output}\n")

    print(f"{written_count} lignes écrites dans {liste_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nettoyer et préparer CommonVoice avec vrais client_id")
    parser.add_argument('cv_root', type=str, help="Chemin vers le dossier racine CommonVoice (ex: db/commonvoice)")
    parser.add_argument('comparisons_file', type=str, help="Chemin vers le fichier comparisons.txt")
    parser.add_argument('output_dir', type=str, help="Répertoire de sortie préparée (ex: data/commonvoice)")
    parser.add_argument('--sampling_frequency', type=int, default=16000, help='Fréquence audio cible (défaut: 16000)')
    parser.add_argument('--vad', action='store_true', help='Appliquer VAD Silero (si disponible)')
    parser.add_argument('--num_jobs', type=int, default=16, help='Nb tâches parallèles (défaut: 16)')
    # Nouvel argument pour spécifier les langues
    parser.add_argument('--langs', nargs='+', default=None,
                        help='Optionnel: Traiter uniquement ces codes langues (sensible à la casse, ex: en fr zh-CN). Traite toutes les langues trouvées si omis.')

    args = parser.parse_args()

    # Vérifier VAD
    if args.vad and not _SILERO_AVAILABLE:
        print("AVERTISSEMENT: --vad demandé mais Silero VAD n'est pas installé. VAD désactivé.")
        args.vad = False
    elif args.vad and model_vad is None:
        # Charger le modèle VAD une seule fois ici si demandé globalement
        print("Chargement du modèle Silero VAD globalement...")
        model_vad = load_silero_vad()

    # Déterminer les langues à traiter
    langs_to_process = []
    all_lang_codes_in_dict = list(LANG_CODE.values()) # Utiliser les valeurs pour la casse correcte

    if args.langs:
        # L'utilisateur a spécifié des langues
        print(f"Traitement demandé pour les langues : {args.langs}")
        for lang in args.langs:
            if lang in all_lang_codes_in_dict:
                langs_to_process.append(lang)
            else:
                # Essayer de trouver une correspondance insensible à la casse comme fallback
                lang_lower = lang.lower()
                found = False
                for code_cased in all_lang_codes_in_dict:
                    if code_cased.lower() == lang_lower:
                         print(f"AVERTISSEMENT: Langue '{lang}' trouvée comme '{code_cased}' (casse corrigée).")
                         langs_to_process.append(code_cased)
                         found = True
                         break
                if not found:
                     print(f"AVERTISSEMENT: Langue '{lang}' spécifiée via --langs inconnue ou non définie dans LANG_CODE. Ignorée.")
        if not langs_to_process:
            print("ERREUR: Aucune des langues spécifiées via --langs n'est valide. Arrêt.")
            exit(1)
    else:
        # Traiter toutes les langues présentes sur le disque qui sont dans LANG_CODE
        print("Aucune langue spécifique demandée. Recherche des langues présentes dans le répertoire source...")
        cv_root_path = Path(args.cv_root)
        for lang_cased in all_lang_codes_in_dict:
            lang_dir = cv_root_path / lang_cased
            if lang_dir.is_dir():
                langs_to_process.append(lang_cased)
        print(f"Langues trouvées à traiter : {langs_to_process}")
        if not langs_to_process:
             print(f"ERREUR: Aucun dossier de langue valide trouvé dans {cv_root_path}. Vérifiez le chemin cv_root.")
             exit(1)


    # --- Boucle Principale sur les langues sélectionnées ---
    print(f"\nLancement de la préparation pour {len(langs_to_process)} langue(s)...")
    for lang_cased in sorted(langs_to_process): # Traiter dans l'ordre pour la lisibilité des logs
        lang_dir = Path(args.cv_root) / lang_cased
        clips_dir = lang_dir / "clips"
        validated_file = lang_dir / "validated.tsv"

        # Vérifier à nouveau au cas où des dossiers seraient incomplets
        if not clips_dir.is_dir() or not validated_file.is_file():
            print(f"⚠️ Skipping {lang_cased}: Dossier 'clips' ou 'validated.tsv' manquant dans {lang_dir}")
            continue

        try:
            clean_and_prepare_commonvoice(
                Path(args.cv_root), Path(args.comparisons_file), lang_cased,
                Path(args.output_dir), args.sampling_frequency, args.vad, args.num_jobs
            )
            print(f"✅ Traitement {lang_cased} terminé.")
        except Exception as e:
            # Afficher l'erreur mais continuer avec les autres langues
            print(f"❌ Erreur CRITIQUE lors du traitement de {lang_cased}: {e}")
            import traceback
            traceback.print_exc() # Afficher la trace complète pour le débogage

    print("\n--- Fin du script de préparation ---")