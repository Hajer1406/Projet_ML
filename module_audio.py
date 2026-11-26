"""Module Audio : Transcription ASR (faster-whisper) -> Prediction de sentiment.

Prend en entree un fichier audio .wav (ou .mp3) et delegue au classifieur textuel.
"""
import io
import sys
from pathlib import Path
from typing import Dict, Any, Union

import av
import numpy as np
from faster_whisper import WhisperModel

from classifieur import predire_texte

RACINE = Path(__file__).resolve().parent
EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
FREQUENCE = 16000  # 16 kHz requis par Whisper
_whisper = None


def trouver_dossier_audios() -> Path:
    """Localise le dossier audios/."""
    for base in (RACINE, RACINE / "dataset_sentiment_ecommerce", RACINE / "data_test_v2"):
        dossier = base / "audios"
        if dossier.is_dir():
            return dossier
    raise FileNotFoundError(f"Dossier 'audios/' introuvable dans {RACINE}")


def _get_whisper():
    """Initialise le modele Whisper en Lazy-loading pour economiser la memoire."""
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper


def charger_audio_pcm(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """Decode le fichier audio vers un tableau float32 mono a 16 kHz."""
    if isinstance(source, (str, Path)):
        source_in = str(source)
    elif isinstance(source, bytes):
        source_in = io.BytesIO(source)
    else:
        source_in = source

    morceaux = []
    with av.open(source_in) as conteneur:
        if not conteneur.streams.audio:
            raise ValueError("Le fichier ne contient aucun flux audio.")
        resampleur = av.AudioResampler(format="s16", layout="mono", rate=FREQUENCE)
        for trame in conteneur.decode(audio=0):
            for t in resampleur.resample(trame):
                morceaux.append(t.to_ndarray().reshape(-1))
        for t in resampleur.resample(None):
            morceaux.append(t.to_ndarray().reshape(-1))

    if not morceaux:
        raise ValueError("Flux audio vide.")
    return np.concatenate(morceaux).astype(np.float32) / 32768.0


def transcrire_audio(source: Union[str, Path, bytes, io.BytesIO], langue: str = "fr") -> str:
    """Transcrit l'audio en texte brut a l'aide de faster-whisper."""
    pcm = charger_audio_pcm(source)
    whisper_model = _get_whisper()
    segments, _ = whisper_model.transcribe(pcm, language=langue, vad_filter=True)
    texte = " ".join(s.text.strip() for s in segments).strip()

    if not texte:
        raise ValueError("La transcription n'a produit aucun texte exploitable.")
    return texte


def predire_audio(source: Union[str, Path, bytes, io.BytesIO]) -> Dict[str, Any]:
    """Pipeline multimodal Audio : Audio (.wav / .mp3) -> ASR -> Classifieur."""
    texte_transcrit = transcrire_audio(source)
    return predire_texte(texte=texte_transcrit, source="audio")


def lister_audios():
    dossier = trouver_dossier_audios()
    fichiers = sorted(p for p in dossier.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier audio trouvé dans {dossier}")
    return fichiers


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--tout":
        cibles = lister_audios()
    elif args:
        cibles = [Path(args[0])]
    else:
        cibles = lister_audios()[:1]
        print(f"Test par défaut avec le premier audio : {cibles[0].name}")

    for chemin in cibles:
        try:
            res = predire_audio(chemin)
            print(f"[{chemin.name}] -> {res}")
        except Exception as e:
            print(f"[{chemin.name}] Erreur : {e}")