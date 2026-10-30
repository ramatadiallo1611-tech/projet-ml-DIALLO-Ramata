# Détection d'attaques réseau par Machine Learning

**Auteur :** Ramata Diallo

Projet 2 (L3) : un modèle apprend à classer chaque flux réseau comme normal (0)
ou attaque (1), à partir de 3 000 exemples.

## Lancer le projet

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Entraîner les modèles depuis la racine du dépôt :

```bash
python src/train.py
```

Le script compare les modèles, puis enregistre le meilleur dans `models/model.joblib`.
La fonction `predict` de `src/predict.py` utilise ensuite ce modèle.

## Résultats

**Nom / Prénom :** Diallo Ramata

Évaluation sur un jeu de validation de 600 flux (20 % des données, dont 167 attaques) :

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|
| Régression logistique | 0,9202 | 0,8982 | 0,9434 |
| Forêt aléatoire | 0,9760 | 0,9760 | 0,9760 |

**Matrices de confusion :**

| Modèle | Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs |
|---|---|---|---|---|
| Régression logistique | 424 | 9 | 17 | 150 |
| Forêt aléatoire | 429 | 4 | 4 | 163 |

**Modèle retenu et pourquoi :** la forêt aléatoire, car elle a le meilleur F1 et le meilleur rappel. Elle rate 4 attaques, contre 17 pour la régression logistique. Elle capte aussi des règles non linéaires, par exemple un `syn_rate` élevé combiné à de nombreuses connexions vers la même machine. Le modèle final a ensuite été ré-entraîné sur les 3 000 flux. Une validation croisée à 5 plis, plus prudente qu'un seul découpage, donne un F1 d'environ 0,96 pour ce modèle.

**Analyse des erreurs (faux négatifs / faux positifs) :** la forêt aléatoire produit 4 faux négatifs, soit 4 attaques ratées sur 167, et 4 faux positifs, soit 4 flux normaux signalés à tort sur 433. Les faux négatifs sont le pire cas en sécurité : une attaque non détectée peut continuer sans qu'aucune alerte ne se déclenche, alors qu'un faux positif coûte surtout un peu de temps de vérification à l'équipe. C'est pourquoi le rappel de la classe attaque compte autant que le F1, et pourquoi l'accuracy seule ne suffit pas avec seulement 28 % d'attaques.
