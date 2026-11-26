"""Nettoyage minimal du texte - partage entre l'entrainement et l'API."""
import re
import unicodedata

# Stop-words francais classiques. Les mots de negation ("ne", "pas", "non", "jamais")
# et d'intensite ("tres", "trop") sont volontairement CONSERVES : ils portent le sentiment.
STOP_WORDS = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "et", "ou", "a",
    "au", "aux", "en", "dans", "sur", "pour", "par", "avec", "ce", "cet", "cette",
    "ces", "c", "mon", "ma", "mes", "son", "sa", "ses", "je", "j", "il", "elle",
    "on", "nous", "vous", "ils", "elles", "me", "m", "se", "s", "qui", "que", "qu",
    "est", "sont", "etait", "ete", "suis", "es", "sommes", "etes", "ai", "as",
    "avons", "avez", "ont", "y", "ca", "n", "t",
}


def _sans_accents(texte: str) -> str:
    """Retire les accents (ex: 'déçu' -> 'decu')."""
    decompose = unicodedata.normalize("NFD", str(texte))
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def nettoyer(texte: str) -> str:
    """Minuscules -> accents retires -> ponctuation retiree -> stop-words filtres."""
    if not texte or not isinstance(texte, str):
        return ""
    texte = texte.lower().replace("’", "'").replace("‘", "'")
    texte = _sans_accents(texte)
    texte = re.sub(r"[^\w\s'€$]", " ", texte)
    texte = texte.replace("'", " ")
    mots = [
        m for m in texte.split()
        if m not in STOP_WORDS and (len(m) > 1 or m in {"€", "$"})
    ]
    return " ".join(mots)


if __name__ == "__main__":
    print(nettoyer("Confirmation de réception #7783 bien réceptionnée."))
    exemple = "Le colis n'est PAS arrivé, très déçu !!!"
    print(f"Original : {exemple}")
    print(f"Nettoyé  : {nettoyer(exemple)}")