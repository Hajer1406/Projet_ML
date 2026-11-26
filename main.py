"""Etape 3 & 4 : Application FastAPI + interface web statique (static/index.html) sur http://127.0.0.1:8000/

Endpoints :
  - GET  /              : Interface Web interactive de test
  - POST /predict/text  : Prediction sur texte brut JSON
  - POST /predict/image : Prediction sur fichier image multipart/form-data
  - POST /predict/audio : Prediction sur fichier audio multipart/form-data
  - POST /predict       : Route avec detection automatique
"""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import ValidationError

from schemas import (FORMATS, MAX_FILE_SIZE_BYTES, FichierEntrant,
                     PredictionResponse, TextRequest)
from classifieur import predire_texte
from module_image import predire_image
from module_audio import predire_audio

RACINE = Path(__file__).resolve().parent

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description="Architecture en cascade (Texte, Image OCR, Audio ASR).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_EXTENSIONS = FORMATS["image"]["extensions"]
AUDIO_EXTENSIONS = FORMATS["audio"]["extensions"]


def valider_fichier(fichier: UploadFile, categorie: str) -> bytes:
    """Lit le fichier puis le valide avec le schema Pydantic FichierEntrant
    (extension, type MIME, taille). Toute erreur -> 422."""
    if not fichier or not fichier.filename:
        raise HTTPException(status_code=422, detail=f"Aucun fichier {categorie} selectionne.")
    # On lit au plus MAX+1 octets : inutile de charger un fichier geant en memoire
    contenu = fichier.file.read(MAX_FILE_SIZE_BYTES + 1)
    try:
        FichierEntrant(
            categorie=categorie,
            nom=fichier.filename,
            type_mime=fichier.content_type,
            taille=len(contenu),
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()[0]["msg"].removeprefix("Value error, "))
    return contenu


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# -----------------------------------------------------------------------------
# Interface Web de test (Etape 4) : fichier statique static/index.html
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def interface_web():
    return FileResponse(RACINE / "static" / "index.html")


# -----------------------------------------------------------------------------
# Routes REST API (Étape 3)
# -----------------------------------------------------------------------------
@app.post("/predict/text", response_model=PredictionResponse, tags=["Predictions"])
def predict_text(payload: TextRequest):
    try:
        return predire_texte(payload.texte, source="texte")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", response_model=PredictionResponse, tags=["Predictions"])
def predict_image(file: UploadFile = File(...)):
    contenu_octets = valider_fichier(file, "image")
    try:
        return predire_image(contenu_octets)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur OCR : {str(e)}")


@app.post("/predict/audio", response_model=PredictionResponse, tags=["Predictions"])
def predict_audio(file: UploadFile = File(...)):
    contenu_octets = valider_fichier(file, "audio")
    try:
        return predire_audio(contenu_octets)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur ASR : {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_auto(
    texte: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Route avec detection automatique du type d'entree."""
    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return predict_image(file=file)
        elif ext in AUDIO_EXTENSIONS:
            return predict_audio(file=file)
        else:
            raise HTTPException(status_code=422, detail=f"Extension '{ext}' non supportee.")
    elif texte is not None:
        try:
            requete = TextRequest(texte=texte)  # meme validation que /predict/text
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()[0]["msg"].removeprefix("Value error, "))
        return predict_text(requete)
    else:
        raise HTTPException(status_code=422, detail="Veuillez fournir un texte ou un fichier valide.")