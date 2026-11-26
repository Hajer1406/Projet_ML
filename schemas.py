"""Schemas Pydantic : validation des requetes (texte + fichiers) et reponse unifiee."""
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 Mo
MAX_TEXT_LENGTH = 5000

# Extensions et types MIME acceptes pour chaque modalite
FORMATS = {
    "image": {
        "extensions": {".png", ".jpg", ".jpeg", ".bmp", ".webp"},
        "mimes": {"image/png", "image/jpeg", "image/bmp", "image/webp"},
    },
    "audio": {
        "extensions": {".wav", ".mp3", ".m4a", ".ogg", ".flac"},
        "mimes": {
            "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3",
            "audio/mp4", "audio/x-m4a", "audio/m4a", "audio/ogg", "audio/flac",
            "audio/x-flac",
        },
    },
}


class TextRequest(BaseModel):
    """Validation de la charge utile JSON pour POST /predict/text."""
    texte: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Texte brut de l'avis client",
        examples=["Le colis est arrivé très vite, produit parfait !"],
    )

    @field_validator("texte")
    @classmethod
    def refuser_texte_vide(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte ne doit pas etre vide.")
        return v


class FichierEntrant(BaseModel):
    """Validation d'un fichier recu en multipart/form-data :
    format (extension), type MIME et taille."""
    categorie: Literal["image", "audio"]
    nom: str = Field(..., min_length=1, description="Nom du fichier envoye")
    type_mime: Optional[str] = Field(None, description="Content-Type declare par le client")
    taille: int = Field(..., description="Taille du fichier en octets")

    @model_validator(mode="after")
    def verifier(self):
        regles = FORMATS[self.categorie]
        ext = Path(self.nom).suffix.lower()
        if ext not in regles["extensions"]:
            raise ValueError(
                f"Format '{ext}' non supporte. Formats autorises : {sorted(regles['extensions'])}"
            )
        mime = (self.type_mime or "").split(";")[0].strip().lower()
        if mime not in regles["mimes"]:
            raise ValueError(
                f"Type MIME '{mime or 'absent'}' non autorise pour {self.categorie}. "
                f"Types autorises : {sorted(regles['mimes'])}"
            )
        if self.taille <= 0:
            raise ValueError("Le fichier envoye est vide.")
        if self.taille > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"Taille excessive ({self.taille / 1024 / 1024:.1f} Mo > "
                f"{MAX_FILE_SIZE_BYTES // 1024 // 1024} Mo)."
            )
        return self


class PredictionResponse(BaseModel):
    """Schema de reponse JSON unifie conforme a l'architecture du projet."""
    source: Literal["texte", "image", "audio"] = Field(
        ..., description="Modalite de la donnee d'entree"
    )
    texte_extrait: str = Field(
        ..., description="Texte brut extrait ou fourni en entree"
    )
    sentiment_predit: Literal["positif", "negatif", "neutre"] = Field(
        ..., description="Classe de sentiment predite"
    )
    confiance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probabilite / score de confiance associe a la prediction",
    )