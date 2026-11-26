"""Chargement du modele texte et prediction unifiee (texte, image, audio)."""
from pathlib import Path
from typing import Dict, Any
import joblib

from nettoyage import nettoyer

RACINE = Path(__file__).resolve().parent
CHEMIN_MODELE = RACINE / "models" / "model.pkl"
CHEMIN_VEC = RACINE / "models" / "vectorizer.pkl"

if not CHEMIN_MODELE.exists() or not CHEMIN_VEC.exists():
    raise FileNotFoundError(
        "Les artefacts 'models/model.pkl' ou 'models/vectorizer.pkl' sont absents. "
        "Veuillez executer 'python entrainement.py' au prealable."
    )

_model = joblib.load(CHEMIN_MODELE)
_vectorizer = joblib.load(CHEMIN_VEC)


def predire_texte(texte: str, source: str = "texte") -> Dict[str, Any]:
    """Prend un texte brut et renvoie la reponse unifiee specifiee dans le sujet :
    {
        "source": "texte" | "image" | "audio",
        "texte_extrait": "...",
        "sentiment_predit": "...",
        "confiance": 0.89
    }
    """
    if not texte or not isinstance(texte, str):
        raise ValueError("Aucun texte fourni pour l'analyse.")

    propre = nettoyer(texte)
    if not propre.strip():
        raise ValueError("Le texte fourni ne contient aucun mot exploitable apres nettoyage.")

    vecteurs = _vectorizer.transform([propre])
    proba = _model.predict_proba(vecteurs)[0]
    idx_max = proba.argmax()

    return {
        "source": source,
        "texte_extrait": texte.strip(),
        "sentiment_predit": str(_model.classes_[idx_max]),
        "confiance": round(float(proba[idx_max]), 2),
    }


if __name__ == "__main__":
    test_phrases = [
        "Livraison super rapide, produit impeccable !",
        "Le colis est arrivé cassé, inadmissible et très déçu.",
        "Article conforme à la commande, emballage standard."
    ]
    for p in test_phrases:
        print(predire_texte(p))