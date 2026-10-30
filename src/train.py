"""Entraîne et compare des modèles de détection d'attaques réseau."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_DONNEES = DOSSIER_PROJET / "data" / "flux_train.csv"
FICHIER_MODELE = DOSSIER_PROJET / "models" / "model.joblib"

CIBLE = "attaque"
COLONNES_CATEGORIELLES = ["protocol"]
COLONNES_NUMERIQUES = [
    "duration",
    "dst_port",
    "src_bytes",
    "dst_bytes",
    "nb_packets",
    "nb_failed_logins",
    "nb_conn_same_host",
    "syn_rate",
]
GRAINE = 42


def explorer_donnees(df):
    """Affiche la taille, les valeurs manquantes et la répartition de la cible."""
    print(f"Taille des données : {df.shape}")
    print(f"Valeurs manquantes :\n{df.isna().sum()}\n")
    print(f"Répartition de la cible :\n{df[CIBLE].value_counts(normalize=True)}\n")


def creer_pretraitement():
    """Impute et met à l'échelle les nombres, encode le protocole."""
    numerique = Pipeline(
        [
            ("imputation", SimpleImputer(strategy="median")),
            ("echelle", StandardScaler()),
        ]
    )
    categoriel = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("numerique", numerique, COLONNES_NUMERIQUES),
            ("categoriel", categoriel, COLONNES_CATEGORIELLES),
        ]
    )


def creer_modeles():
    """Retourne les modèles à comparer, chacun avec son prétraitement."""
    classifieurs = {
        "Régression logistique": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=GRAINE
        ),
    }
    return {
        nom: Pipeline([("pretraitement", creer_pretraitement()), ("modele", classifieur)])
        for nom, classifieur in classifieurs.items()
    }


def evaluer_modele(nom, modele, X_val, y_val):
    """Affiche les métriques du modèle et retourne son F1 sur la classe attaque."""
    predictions = modele.predict(X_val)
    f1 = f1_score(y_val, predictions)
    vrais_neg, faux_pos, faux_neg, vrais_pos = confusion_matrix(y_val, predictions).ravel()

    print(f"=== {nom} ===")
    print(classification_report(y_val, predictions, target_names=["normal", "attaque"], digits=4))
    print(f"F1 (attaque)        : {f1:.4f}")
    print(f"Rappel (attaque)    : {recall_score(y_val, predictions):.4f}")
    print(f"Précision (attaque) : {precision_score(y_val, predictions):.4f}")
    print(f"Matrice de confusion : VN={vrais_neg} FP={faux_pos} FN={faux_neg} VP={vrais_pos}\n")
    return f1


def main():
    df = pd.read_csv(FICHIER_DONNEES)
    explorer_donnees(df)

    X = df.drop(columns=CIBLE)
    y = df[CIBLE]
    # stratify garde la même proportion d'attaques dans les deux jeux
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=GRAINE
    )

    scores = {}
    modeles = creer_modeles()
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        scores[nom] = evaluer_modele(nom, modele, X_val, y_val)

    meilleur_nom = max(scores, key=scores.get)
    print(f"Modèle retenu : {meilleur_nom} (F1 = {scores[meilleur_nom]:.4f})")

    # Le modèle final apprend sur toutes les données avant d'être enregistré
    meilleur_modele = creer_modeles()[meilleur_nom]
    meilleur_modele.fit(X, y)
    FICHIER_MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(meilleur_modele, FICHIER_MODELE, compress=3)
    print(f"Modèle enregistré dans {FICHIER_MODELE.relative_to(DOSSIER_PROJET)}")


if __name__ == "__main__":
    main()
