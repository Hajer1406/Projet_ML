"""Etape 1 - Modelisation et entrainement du classifieur de sentiment texte.

Enonce :
  - Charger textes.csv
  - Nettoyer le texte (minuscules, stop-words...)
  - Vectoriser (TF-IDF) et entrainer un classifieur supervise (Regression Logistique)
  - Evaluer (rapport de classification + matrice de confusion)
  - Sauvegarder les artefacts (model.pkl, vectorizer.pkl)

Usage   : python entrainement.py
Produit : models/model.pkl, models/vectorizer.pkl,
          models/matrice_confusion.png, models/metriques.json
"""
import json
import unicodedata
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

from nettoyage import nettoyer

RACINE = Path(__file__).resolve().parent
CSV_CANDIDATS = [
    RACINE / "dataset_sentiment_ecommerce" / "textes.csv",
    RACINE / "data_test_v2" / "textes.csv",
    RACINE / "textes.csv",
]
CSV = next((p for p in CSV_CANDIDATS if p.exists()), CSV_CANDIDATS[0])
SORTIE = RACINE / "models"
CLASSES = ["negatif", "neutre", "positif"]
SEED = 42

NOMS_TEXTE = ["texte", "text", "avis", "review", "commentaire", "comment"]
NOMS_LABEL = ["sentiment_ref", "sentiment", "label", "classe", "class"]

SYNONYMES = {
    "negatif": "negatif", "negative": "negatif", "neg": "negatif",
    "neutre": "neutre", "neutral": "neutre", "neutr": "neutre",
    "positif": "positif", "positive": "positif", "pos": "positif",
}


def charger_csv(chemin: Path) -> pd.DataFrame:
    if not chemin.exists():
        raise SystemExit(f"Fichier introuvable : {chemin}")
    for encodage in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(chemin, encoding=encodage, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise SystemExit("Encodage du fichier CSV non reconnu.")


def trouver_colonne(df: pd.DataFrame, noms: list, role: str) -> str:
    colonnes = {c.strip().lower(): c for c in df.columns}
    for nom in noms:
        if nom in colonnes:
            return colonnes[nom]
    raise SystemExit(f"Colonne pour '{role}' introuvable. Colonnes existantes : {list(df.columns)}")


def normaliser_label(valeur) -> str:
    s = unicodedata.normalize("NFD", str(valeur).strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return SYNONYMES.get(s, s)


# -------------------------------------------------- 1. Chargement & preparation
df = charger_csv(CSV)
col_texte = trouver_colonne(df, NOMS_TEXTE, "texte")
col_label = trouver_colonne(df, NOMS_LABEL, "sentiment")
df = df[[col_texte, col_label]].rename(columns={col_texte: "texte", col_label: "sentiment_ref"})
df = df.dropna(subset=["texte", "sentiment_ref"]).drop_duplicates("texte")
df["sentiment_ref"] = df["sentiment_ref"].apply(normaliser_label)

inconnus = df[~df["sentiment_ref"].isin(CLASSES)]
if len(inconnus):
    print(f"Lignes ignorees (classe inconnue) : {len(inconnus)}")
    df = df[df["sentiment_ref"].isin(CLASSES)]

print(f"{len(df)} avis charges depuis : {CSV.name}")
print(df["sentiment_ref"].value_counts().to_string(), "\n")

# -------------------------------------------------- 2. Nettoyage
df["texte_propre"] = df["texte"].apply(nettoyer)
df = df[df["texte_propre"].str.strip() != ""]

X, y = df["texte_propre"], df["sentiment_ref"]
effectif_min = int(y.value_counts().min())
if effectif_min < 2 or y.nunique() < 3:
    raise SystemExit("Le dataset doit contenir au moins 2 exemples pour chacune des 3 classes.")

# -------------------------------------------------- 3. Selection du modele (TF-IDF + LogReg)
CANDIDATS = {
    "TF-IDF mots (1,2) + LogReg (C=5)": (
        lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        lambda: LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=SEED),
    ),
    "TF-IDF mots (1,2) + LogReg (C=10)": (
        lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        lambda: LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=SEED),
    ),
    "TF-IDF chars (2,5) + LogReg": (
        lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        lambda: LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced", random_state=SEED),
    ),
}

n_plis = min(5, effectif_min)
n_reps = 10 if len(df) < 100 else 3
cv = RepeatedStratifiedKFold(n_splits=n_plis, n_repeats=n_reps, random_state=SEED)
print(f"Validation croisee ({n_plis} plis x {n_reps} repetitions, F1-macro) :")

resultats_cv = {}
for nom, (fabrique_vec, fabrique_mod) in CANDIDATS.items():
    pipe = make_pipeline(fabrique_vec(), fabrique_mod())
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    resultats_cv[nom] = {
        "moyenne": round(float(scores.mean()), 3),
        "ecart_type": round(float(scores.std()), 3),
    }
    print(f"  {nom:<42} : {scores.mean():.3f} +/- {scores.std():.3f}")

meilleur_nom = max(resultats_cv, key=lambda k: resultats_cv[k]["moyenne"])
fabrique_vec, fabrique_mod = CANDIDATS[meilleur_nom]
print(f"\nModele retenu : {meilleur_nom}")

# -------------------------------------------------- 4. Evaluation sur jeu de test (hold-out)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)
print(f"Train : {len(X_train)} avis | Test : {len(X_test)} avis")

vectorizer_eval = fabrique_vec()
model_eval = fabrique_mod()
model_eval.fit(vectorizer_eval.fit_transform(X_train), y_train)
y_pred = model_eval.predict(vectorizer_eval.transform(X_test))

print("\n=== Rapport de classification (jeu de test) ===")
print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
print("Matrice de confusion (lignes = verite, colonnes = prediction) :")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES), "\n")

SORTIE.mkdir(exist_ok=True, parents=True)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", colorbar=False)
plt.title(f"Matrice de Confusion ({meilleur_nom})")
plt.tight_layout()
plt.savefig(SORTIE / "matrice_confusion.png", dpi=150)
plt.close()

# -------------------------------------------------- 5. Entrainement final & sauvegarde
vectorizer_final = fabrique_vec()
model_final = fabrique_mod()
model_final.fit(vectorizer_final.fit_transform(X), y)

joblib.dump(model_final, SORTIE / "model.pkl")
joblib.dump(vectorizer_final, SORTIE / "vectorizer.pkl")

metriques = {
    "source_dataset": str(CSV.relative_to(RACINE) if CSV.is_relative_to(RACINE) else CSV.name),
    "nb_echantillons": int(len(df)),
    "modele_selectionne": meilleur_nom,
    "validation_croisee": resultats_cv,
    "test_classification_report": classification_report(
        y_test, y_pred, labels=CLASSES, zero_division=0, output_dict=True
    ),
    "matrice_confusion": cm.tolist(),
    "classes": [str(c) for c in model_final.classes_],
}
with open(SORTIE / "metriques.json", "w", encoding="utf-8") as f:
    json.dump(metriques, f, ensure_ascii=False, indent=2)

print(f"Artefacts sauvegardes avec succes dans : {SORTIE}\n")

# -------------------------------------------------- 6. Test rapide de prediction
print("=== Test rapide de prediction ===")
phrases_test = [
    "Livraison rapide, je recommande !",
    "Le produit ne marche pas, très déçu",
    "Le colis est arrivé cassé, inadmissible et très déçu.",
    "Colis reçu conforme à la commande",
]
for phrase in phrases_test:
    v = vectorizer_final.transform([nettoyer(phrase)])
    p = model_final.predict_proba(v)[0]
    print(f"{phrase!r:<55} -> {model_final.classes_[p.argmax()]} ({p.max():.2f})")