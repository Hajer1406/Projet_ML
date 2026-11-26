"""Module Image : Extraction OCR (pytesseract) -> Prediction de sentiment.

Prend en entree une image (chemin, fichier ou octets en memoire)
et delegue au classifieur textuel.
"""
import io
import re
import sys
from pathlib import Path
from typing import Dict, Any, Union

import pytesseract
from PIL import Image, ImageOps

from classifieur import predire_texte

RACINE = Path(__file__).resolve().parent
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Configuration Tesseract sous Windows (si present dans le chemin par defaut)
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)


def trouver_dossier_images() -> Path:
    """Localise le dossier images/."""
    for base in (RACINE, RACINE / "dataset_sentiment_ecommerce", RACINE / "data_test_v2"):
        dossier = base / "images"
        if dossier.is_dir():
            return dossier
    raise FileNotFoundError(f"Dossier 'images/' introuvable dans {RACINE}")


def extraire_texte_image(source: Union[str, Path, bytes, io.BytesIO], langue: str = "fra") -> str:
    """Extrait le texte d'une image en supprimant le bruit d'en-tete et les etoiles."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    try:
        img = Image.open(source)
    except Exception as e:
        raise ValueError(f"Impossible d'ouvrir l'image : {e}")

    # Niveaux de gris et orientation EXIF
    img = ImageOps.exif_transpose(img).convert("L")

    # Agrandissement pour ameliorer la qualite OCR
    if img.width < 1000:
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.BILINEAR)

    texte_brut = pytesseract.image_to_string(img, lang=langue)

    # Filtrage des lignes de bruit : bandeaux, extensions de fichiers, etoiles OCR bruitées
    lignes_valides = []
    pattern_extension = re.compile(r"\b[a-zA-Z0-9_\-]+\.(png|jpe?g)\b", re.IGNORECASE)

    for ligne in texte_brut.splitlines():
        l = ligne.strip()
        if not l:
            continue
        # Ignorer les bandeaux type "CAPTURE AVIS CLIENT", "AVIS CLIENT VERIFIE", etc.
        if re.search(r"\b(avis client|client verifie|capture avis)\b", l, re.IGNORECASE):
            continue
        # Ignorer les noms de fichiers
        if pattern_extension.search(l):
            continue
        # Ignorer les lignes composées uniquement de bruit ou symboles d'étoiles OCR (ex: 'x k%k x x', '***')
        caracteres_utiles = re.sub(r"[\s%kx#*—\-_|/\\;:!?.]", "", l, flags=re.IGNORECASE)
        if len(caracteres_utiles) < 3:
            continue

        lignes_valides.append(l)

    texte_final = "\n".join(lignes_valides).strip()

    # Repli de securite si tout a ete filtre
    if not texte_final:
        texte_final = texte_brut.strip()

    if not texte_final:
        raise ValueError("L'OCR n'a extrait aucun texte exploitable depuis l'image.")

    return texte_final


def predire_image(source: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
    """Pipeline multimodal Image : Image -> OCR -> Classifieur de sentiment."""
    texte_extrait = extraire_texte_image(source)
    return predire_texte(texte=texte_extrait, source="image")


def lister_images():
    dossier = trouver_dossier_images()
    images = sorted(p for p in dossier.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {dossier}")
    return images


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--tout":
        cibles = lister_images()
    elif args:
        cibles = [Path(args[0])]
    else:
        cibles = lister_images()[:1]
        print(f"Test par défaut avec la première image : {cibles[0].name}")

    for chemin in cibles:
        try:
            res = predire_image(chemin)
            print(f"[{chemin.name}] -> {res}")
        except Exception as e:
            print(f"[{chemin.name}] Erreur : {e}")