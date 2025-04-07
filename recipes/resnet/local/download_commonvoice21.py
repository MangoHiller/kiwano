#!/usr/bin/env python3

import logging
import hashlib
import tarfile
import argparse
import csv
from pathlib import Path
from urllib.request import urlretrieve
from kiwano.utils import Pathlike, urlretrieve_progress

# Lien vers le fichier comparisons.txt
#COMPARISONS_URL = "https://cloud.ovgu.de/s/MGXi8ijXHSpsjEc/download?path=%2F&files=comparisons.txt&downloadStartSecret=fzs2x3x7p3i"

# Dictionnaire à compléter manuellement les liens sont disponibles sur le site de commonvoice
COMMONVOICE_URLS_HASHES = {


    "fy-nl": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-fy-NL.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083140Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7898094f416a615992e2f35a4f67cbfeccaeb3e3e1a85c1ef93699e1d708af71d9e54726398c8023f5cf73a295dc7f71832e7d198c9bf6b76736f1adcbf91019f53ea544ab9ed48b272b915320afbff4dbca90177816ae6418e61766d8dee54391b08561fd89ade27cf6d57bea37364723e389a50f12ea7900cd9344bf30f59c5f70ff2d89105c0154fd857e45b0b4e492e23eb487983dcab0529e3fbe8c3347dcfcfed01bbb9efc24eb132cbab992921856313d78ffec846cbf01a02dcf0c4e3270e2edf853d66746587f9e23701bdacf4624e2b77c4ef65903af56f757c385ef2b56bdc48f914d957d52415ca21a14306bed73876eca7c5fdb0824bf1c58c",
        "sha256": "287fe058384fdd7a933a6c2618eeb28bb139c4e6165985670aeda47fddb79e4b",
        "langue": "Frisian"
    },
    
    "ga-ie": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-ga-IE.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083218Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=7baee59cc4df8051d52adc12c7a71fe1de01bf3add59b4b9ef1589a435dfc2c6dee46db1f6ecb68ca4d8fd8a1c3e4179b1ea2a741e3dd21891e83a9d4dcf66f4254c1866593c81db6dff89cf5f5c9cbf7d3f594191c74818400dddc7ae9c2b62f6b565ff61043c48b69dcb544377fabfb36debf215b98a896f0c6ebfa7cdef8e538ede4409a7f27e17f55104185d7b6bd246be67075d95db049a94a5019ad02a18e6058a7456c8484fa8a2d8811346d25b049837e81b0fcca06e5820640ef75d0549feb72442a9d4fef21947bd8c46a740e252166a10ce4dae279e19628e4b2869326cda5200b55a295c71ee20120c31099d17ba21655be4cd3bcc02ba10bf19",
        "sha256": "20c1cfc6ecf9adf3a652550152c9c6eff8fd53e0d50ee35191aae25073798c55",
        "langue": "Irish"
    },
      
        
    "hy-am": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-hy-AM.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083245Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=55288e122e2874ef8a59c7acea8164bce02cb8395aff27feebfc8f886671414937ea2cbec4f8b328d4fd20bafc69da09e8ff67d71ee64808e700157a8b97a79286b4c2b3d98b9f8a7a8f0497424adef69a3e65d06271eaa6b3fb28dbb96c84981701a14750c29c6cb2b5d3be10415f208341ffa60ea19716eacb23265c1b6ad99505e216fbf82629ab407b98f773b0c43838a210e7c4e7bed1a509cae91278e505aa8005530c2ca180fcf4fece1a08e5ca38ebf09019ec29672b102a4c3a5601ff4ff8249267c48be13687f0946aefc38977010a68d301a2127c8a565fdc456b8109b923593b1d491689cdf1e6c26a44ba79a76a5b297ab6e1a49fb5e1ba65d9",
        "sha256": "6773f9a6037ef1f08898c37e20ba29f07b1ce8a3000087734d4ad940450d7297",
        "langue": "Armenian"
    },
       
    
    "ne-np": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-ne-NP.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083311Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=87eba84232807c38337831c87a18af15dabe7da545b5d4411f38d9baa9324d5b914bc73b7f307464f8729efb345b603d7e2af9a0abb528f7f508fc216a3adbb321ee5c9d271e3d08adfecbcf05591f42ffa1af6aa0395db25195b90892e15680e0ac289ad1bb530f0d0e349f9c898daa5fcdcb472b17d7e2675e690f3993e581b84b55c8ce89f6e4d9ff9e59ee660c30fc01aa3ec2dfb4c82116c08e06151fd087e7e166fa0ef4ba2e88a95f6f4a308a1ce9bcc84aa8c6426cec8046c9f9ea6a3aa80865850415e20f3112a810f4c4878311075c4acacb8e910369d63fdc59e9d7892091c6dd7d67755e7ecc5afe99ef6ad42674d6d2f872c91cfc77fdcaa3bf",
        "sha256": "bcc4230ea3c5777b83f8e0d598ab8848b108c31d417a499abbfe335947f5c94f",
        "langue": "Nepali"
    },
    

    "nn-no": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-nn-NO.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083341Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=546d7cf8ba33ad98cb44d6a747995b154b3af73ffc179b48e51229a7814aafb44874c0f8647d3efc7646473337bc3b719b5a62fc0857704c590f8e77229cfb25044a1b595ec2f3e1de4fc6702a98037c29081e98983da1ededd8c4bbc0382340c575ae6005c9d60dd9b365776d04752b968746652bf19f9503acdce2e6f06ae05a1ab4e2e5da90ddcca555fd4489e948563a546e67bf87846dd9341ff5b88ff0a5f66ce3f1e4b1e588408853f8000b61a19c85c0341c8cae472cf3a8140764cd787a8bf57fedfaefd30eb605f1949a09c8a4d0e0f083a8ec006de30414a1b290340d56ba88892db89a4c1ea0df919dc921c2081123251854324a02c632122cc8",
        "sha256": "955a49e3001a03c781ba40335cf5548b16501d553b3f28c02ddda8c446506ec0",
        "langue": "NorwegianNynorsk"

    },
        
    
    "pa-in": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-pa-IN.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083410Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=2d56bcd40a68230840e0435e3ebf10dd6671738673d65b84eb81bad82668313178357f764d3f4e9326038b91ad4c903f956eab6a8cb59cd37056a26d6bd6ae65d61e721caff03a22a309513dde484f0b0da8cb434b41c3bf8be57663c816d652a37b0589716818f9d2d759dc64aa8d9fb32067609598c679028992858d542125a6662f9d48e2b18bac2d74b438a328d7119c25642089e901c910c6a59540a93194ca9e692236b636dfce7ddb25ae6cac5c376c7788da2bc441bda8a58c30fbea47299d079103b37534143463379ea4063028f0b78cd5bfd1ac5b93e2e69d7c73d8dcd48c1fbc05b898a557f000c1844cc29d247da3b639bd6ac11084490c50dd",
        "sha256": "6f7c8358d6eaf0bae2ec8c2b6dc37b5ad870798f0860b59117fa3d0edaae592b",
        "langue": "Punjabi"
    },

    
    "sv-se": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-sv-SE.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083437Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=12c9a8a64b3102380a21124d9da26bca3ab667de8147d454518ce7eb62d0a2fc66ef6a534c116d1262c4a8f4bf0c790e818c3ecf7c74e3598562937ad29e8d217608b9f115cfbe2c5841026b061c25fabbf8e37c00b7d18f4933b1cc544e36e0220fe84f1bfaca18402b9a5856f10a4a12c8228cc5cfff1f65bc3eb3347dc2fdc826963659aad3be9614c7cc5fb3884bc4dff0e609f62970904be54cfe8e130a20b1ed53bf94baa3b3ff1a1e47e844a1797fd246fc03229b87abcc1592cc9122d19280700a6b1eebbe48e66558eea2123ff6562980afb57682c963ca2ceda56f98509174fd7f4023fa69193a52b1e810746f0628e40db81ca7fbafb8fb6081e2",
        "sha256": "aeb753e62cca50d3ec178840b87b096f10756563e66f936d19c55d758a89af44",
        "langue": "Swedish"
    },
        
    
    "zh-cn": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-zh-CN.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083503Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7c20b3e08a7445225117c940e51b17deb49cd39ea356324bd349e2d8826d73542693a0bcb9e8ef17b1d837b6501f5d8075d080eff1b9f8ed13472642cbc32a979c1cce3567377c68ab1406c970917ce38e12baf77f0a0b4eee3d6aec0e156c66af1755035b4719fed43f0eab8f0caada87675a5b379cbf8b76eed0b0336f53d2f8d395d2d47b38c8412af018d002e2946c020f9bb1b7501be287f72c4205742096dbcfc58713ed8ee41b95c3e764ee756124bae90183bb01195105b7543d08f7bdc6731eb16e672a785f1b0a28ad22ee238c9ba482e7427a2d64c1a9ed52da56273c0f0ccd902614ae741cae1b5066676ee736da1fb919fc23874608035287c",
        "sha256": "b30f2ab0e3afd88ff93974ba7e24091955f150c49486293e1dcda52439ce587a",
        "langue": "Chinois-Chine"
    },
    
    "zh-hk": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-zh-HK.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083526Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=3078c193178728f96462b92f24032523f189efea8bbc521848e6b467066117b63e5a76393e6e302f816ccbe22737666822ec318511d333ba708e274d4dc81fbd97fcc40a9be90df69a7c578652dfb7120536dfc3720081109a468f32c24d0b41ee823aa1ff7abaab7406dd964fcec6d7e47cb2de57ecf33e375e46cf9f63f431ce365d3628d89aa9d75ff9e4b76139a646d5ca45e475118a983bfc392b51704834981d0a6c166adfc578951b67bdf30cb852bb264218722f925eaffad7522e0d67c67ba014b5acd37863bb55809f8a9a032818c0ffa39bd64519f6e5d4610a635871e8314ab56bf78916d39fe1a448512b9126d830393792d3c8627e27373c1b",
        "sha256": "4bc64b2db61c3bf8451d9e45082fcb43d88a184c8f82a9ea93f14abfd6b7cc70",
        "langue": "Chinois-Hk"
    },

    "zh-tw": {
        "url": "https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-zh-TW.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250403%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250403T083550Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=2f1f9ab7f1fa151a16e7c74b45afd5709609a26363e3d5c610e464bc431b3e5b7e4c8401274f2aad68dc8c57070c02d76ab7bf403ac449913a30a373ac353b6cc49a2c56ce9a9fdc4d9f2bc1a23bd48d3f68af6915500fdd356bed3182f7949b2bf7895c848ffcd2cc65196557a0a39abaa637c87b4402f80d07a6689c9a297a9a74d90e2bb0744d2d0b1b1efa4f0f2d0242df579d5136395e90dd40105f8f6ce01a0a982af6498636b08ed8710ebf14a22393c9bb9320856f228cf837963e5d2074e35cc4ee7f8a0e1f44ce2074fd089c1c62f43145d5eb1524a3b748e21b106ba3c892b9f3b63901f46d2e70bc7a4cd72396b0957a73cb52350ef81b6e5a89",
        "sha256": "1d8550f01215fb1e0ed8a1adac93b9a1551e212e1e74cf421f97708476fa8a37",
        "langue": "Chinois-TW"
    }, 
}

"""def download_comparisons_file(output_path: Path):
    if not output_path.exists():
        logging.info("Téléchargement de comparisons.txt...")
        urlretrieve(COMPARISONS_URL, output_path)
        logging.info("Téléchargement terminé.")
    else:
        logging.info("comparisons.txt déjà présent, téléchargement ignoré.")"""

def extract_languages_from_comparisons(file_path: Pathlike) -> set:
    langs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ['audio1', 'audio2']:
                value = row[col]
                if value and '@' in value:
                    lang_code = value.split('@')[0]
                    langs.add(lang_code.lower())
    return langs

def verify_sha256(filename: Path, expected_hash: str):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def download_and_extract(lang_code: str, url: str, expected_hash: str, output_dir: Path, archive_dir: Path, force_download: bool):
    tar_name = f"{lang_code}.tar.gz"
    tar_path = archive_dir / tar_name
    lang_out_dir = output_dir / lang_code
    lang_out_dir.mkdir(parents=True, exist_ok=True)

    if tar_path.exists() and not force_download:
        logging.info(f"{tar_name} déjà présent, téléchargement ignoré.")
    else:
        logging.info(f"Téléchargement : {tar_name}")
        urlretrieve_progress(url, filename=tar_path, desc=f"Téléchargement {lang_code}")

    logging.info(f"Vérification SHA256 pour {lang_code}")
    if not verify_sha256(tar_path, expected_hash):
        logging.error(f"SHA256 incorrect pour {lang_code}, suppression.")
        tar_path.unlink(missing_ok=True)
        raise ValueError(f"SHA256 incorrect pour {lang_code}")

    if (lang_out_dir / "clips").exists() and not force_download:
        logging.info(f"{lang_code} déjà extrait, extraction ignorée.")
        return

    logging.info(f"Extraction propre de {tar_name}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            try:
                lang_index = member_path.parts.index(lang_code)
            except ValueError:
                continue
            subpath = Path(*member_path.parts[lang_index + 1:])
            if subpath.parts and (subpath.parts[0] == "clips" or subpath.name == "validated.tsv"):
                member.name = str(subpath)
                tar.extract(member, path=lang_out_dir)

    logging.info(f"Extraction terminée : {lang_code}")

def download_commonvoice(target_dir: Pathlike, force_download: bool = False):
    target_dir = Path(target_dir)
    comparisons_path = target_dir / "comparisons.txt"

    if not comparisons_path.exists():
        raise FileNotFoundError(f"comparisons.txt introuvable dans {comparisons_path}. Télécharge-le manuellement avant d’exécuter ce script.")


    langs_needed = extract_languages_from_comparisons(comparisons_path)
    logging.info(f"Langues détectées : {sorted(langs_needed)}")

    audio_dir = target_dir
    archive_dir = target_dir / "archives"
    audio_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    for lang_code in langs_needed:
        if lang_code not in COMMONVOICE_URLS_HASHES:
            logging.warning(f"Aucune URL pour la langue '{lang_code}', ignorée.")
            continue

        infos = COMMONVOICE_URLS_HASHES[lang_code]
        download_and_extract(lang_code, infos["url"], infos["sha256"], audio_dir, archive_dir, force_download)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Téléchargement structuré de CommonVoice pour Kiwano.")
    parser.add_argument('--force_download', action='store_true', default=False,
                        help="Force le téléchargement même si les fichiers existent déjà.")
    parser.add_argument('target_dir', type=str, help='Répertoire de destination (ex: db/commonvoice)')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    download_commonvoice(args.target_dir, args.force_download)
