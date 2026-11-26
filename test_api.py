"""Etape 5 - Tests automatises : unitaires + integration (pytest + TestClient httpx).

Lancer depuis la racine du projet :  pytest -v
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from nettoyage import nettoyer
from schemas import FichierEntrant, TextRequest

client = TestClient(main.app)


def verifier_structure(reponse, source):
    """200 OK + structure JSON attendue."""
    assert reponse.status_code == 200
    data = reponse.json()
    assert set(data) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert data["source"] == source
    assert data["sentiment_predit"] in {"positif", "negatif", "neutre"}
    assert 0.0 <= data["confiance"] <= 1.0


# =============================== TESTS UNITAIRES ================================
def test_nettoyer():
    assert nettoyer("Le colis est ARRIVÉ !") == "colis arrive"


def test_schema_texte_valide():
    assert TextRequest(texte="Super produit").texte == "Super produit"


def test_schema_texte_invalide():
    with pytest.raises(ValidationError):
        TextRequest(texte="")


def test_schema_fichier_valide():
    FichierEntrant(categorie="image", nom="avis.png", type_mime="image/png", taille=100)


def test_schema_fichier_invalide():
    with pytest.raises(ValidationError):
        FichierEntrant(categorie="image", nom="avis.txt", type_mime="text/plain", taille=100)


# ============================ TESTS D'INTEGRATION ===============================
# ------------------------------ POST /predict/text ------------------------------
def test_predict_text_200():
    r = client.post("/predict/text", json={"texte": "Livraison rapide, je recommande !"})
    verifier_structure(r, "texte")


def test_predict_text_422():
    assert client.post("/predict/text", json={}).status_code == 422
    assert client.post("/predict/text", json={"texte": ""}).status_code == 422


# ----------------------------- POST /predict/image ------------------------------
def test_predict_image_200(monkeypatch):
    monkeypatch.setattr(main, "predire_image", lambda _: {
        "source": "image", "texte_extrait": "texte simule", "sentiment_predit": "neutre", "confiance": 0.5})
    r = client.post("/predict/image", files={"file": ("avis.png", b"octets", "image/png")})
    verifier_structure(r, "image")


def test_predict_image_422():
    r = client.post("/predict/image", files={"file": ("avis.txt", b"octets", "text/plain")})
    assert r.status_code == 422


# ----------------------------- POST /predict/audio ------------------------------
def test_predict_audio_200(monkeypatch):
    monkeypatch.setattr(main, "predire_audio", lambda _: {
        "source": "audio", "texte_extrait": "texte simule", "sentiment_predit": "neutre", "confiance": 0.5})
    r = client.post("/predict/audio", files={"file": ("avis.mp3", b"octets", "audio/mpeg")})
    verifier_structure(r, "audio")


def test_predict_audio_422():
    r = client.post("/predict/audio", files={"file": ("avis.txt", b"octets", "text/plain")})
    assert r.status_code == 422


# --------------------- POST /predict (detection automatique) ---------------------
def test_predict_auto_200():
    r = client.post("/predict", data={"texte": "Colis reçu conforme à la commande."})
    verifier_structure(r, "texte")


def test_predict_auto_422():
    assert client.post("/predict").status_code == 422