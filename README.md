# Classifieur de Sentiment Multi-source (E-commerce Customer Voice)

Prédit le sentiment (positif, négatif ou neutre) d'un avis client reçu en **texte**, **image** (OCR) ou **audio** (transcription), via une API REST FastAPI et une mini-application web de test.

## 1. Installation des dépendances

1. Créer et activer un environnement virtuel :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   (Linux / macOS : `source .venv/bin/activate`)

2. Installer les dépendances Python listées dans `requirements.txt` :
   ```bash
   pip install -r requirements.txt
   ```

3. Installer deux programmes externes (hors `pip`) :
   - **Tesseract OCR**, avec le pack de langue française (`fra`), utilisé par `pytesseract`.
   - **FFmpeg**, utilisé par Whisper pour lire les fichiers audio.

4. Entraîner le modèle (génère `models/model.pkl` et `models/vectorizer.pkl`) :
   ```bash
   python entrainement.py
   ```

## 2. Lancement de l'API et de la mini-application web

**Lancer l'API :**
```bash
uvicorn main:app --reload
```

**Mini-application web de test** (HTML/JS, servie par l'API, aucune commande supplémentaire) : ouvrir http://127.0.0.1:8000/ dans le navigateur.

**Documentation interactive Swagger** : http://127.0.0.1:8000/docs (ReDoc : http://127.0.0.1:8000/redoc).

**Lancer les tests automatisés :**
```bash
pytest -v
```

## 3. Captures d'écran

### Application web en fonctionnement
![Application web](docs/)

### Documentation interactive Swagger
![Swagger](docs/)
## Choix techniques
 
- **TF-IDF + Régression Logistique** : méthode simple, rapide, interprétable, adaptée à un petit dataset ; `class_weight="balanced"` pour équilibrer les classes ; `predict_proba` fournit l'indice de confiance.
- **Nettoyage minimal partagé** (`nettoyage.py`) : minuscules, accents et ponctuation retirés, stop-words filtrés. Les négations (`ne`, `pas`) et les mots d'intensité (`très`) sont conservés car ils portent le sentiment. Le même code sert à l'entraînement et à l'API, pour éviter tout décalage.
- **Architecture en cascade** : image et audio sont convertis en texte, puis un seul classifieur décide.
- **Interface HTML/JS statique** servie par FastAPI : aucun framework supplémentaire.
- **Tests** : unitaires (nettoyage, schémas) et d'intégration (4 endpoints, réponses 200 et 422). L'OCR et la transcription y sont simulés (`monkeypatch`) pour que les tests soient rapides et indépendants de Tesseract et Whisper.
## Limites
 
Le dataset fourni est très petit (24 avis, 8 par classe) : les performances (voir `models/metriques.json`) sont limitées et les phrases éloignées du vocabulaire du dataset peuvent être mal classées. Pistes d'amélioration : enrichir le dataset, tester un SVM ou des embeddings.
