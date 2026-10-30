"""Prédit si des flux réseau sont normaux (0) ou des attaques (1)."""

from pathlib import Path

import joblib
import pandas as pd

FICHIER_MODELE = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def predict(df: pd.DataFrame) -> list:
    """Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    modele = joblib.load(FICHIER_MODELE)
    return [int(prediction) for prediction in modele.predict(df)]
