# HEART FAILURE PROJECT
Ce projet consiste à la prédiction de la mortalité liée à l'insuffisance cardiaque à partir de données cliniques.
# 🫀 PRÉDICTION D'INSUFFISANCE CARDIAQUE — Application d'Aide à la Décision Médicale

> MEDICAL DECISION SUPPORT APPLICATION PREDICTING HEART FAILURE RISK WITH EXPLAINABLE ML (SHAP) 
> Coding Week · Ecole Centrale Casablanca · 

---

## 📋 Table des Matières

1. [objectif du projet](#objectif-du-projet)
2. [Jeu de données](#jeu-de-données)
3. [Structure du projet](#structure-du-projet)
4. [Installation & Configuration](#installation--configuration)
5. [Étape 1 — Traitement des données](#étape-1--traitement-des-données)
6. [Étape 2 — Entraînement des modèles](#étape-2--entraînement-des-modèles)
7. [Étape 3 — Évaluation des modèles](#étape-3--évaluation-des-modèles)
8. [Étape 4 — Explicabilité SHAP](#étape-4--explicabilité-shap)
9. [Étape 5 — Interface Web (Streamlit)](#étape-5--interface-web-streamlit)
10. [Étape 6 — Tests automatisés](#étape-6--tests-automatisés)
11. [Étape 7 — Pipeline CI/CD (GitHub Actions)](#étape-7--pipeline-cicd-github-actions)
13. [Réponses des Questions critiques ](#réponses-des-questions-critiques)
14. [Documentation Prompt Engineering](#documentation-prompt-engineering)
15. [Résumé pour faire fonctionner le projet](#résumé-pour-faire-fonctionner-le-projet)

---

##  objectif du projet

Ce projet est un outil avancé d'aide à la décision clinique conçu pour aider les médecins à prédire le risque d'insuffisance cardiaque chez les patients.

---

## Jeu de données

**Source :** [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords)

| Caractéristique | Description |
|---|---|
| `age` | Âge du patient |
| `anaemia` | Diminution des globules rouges |
| `creatinine_phosphokinase` | Taux de l'enzyme CPK dans le sang |
| `diabetes` | Si le patient est diabétique |
| `ejection_fraction` | Pourcentage de sang quittant le cœur |
| `high_blood_pressure` | Si le patient a de l'hypertension |
| `platelets` | Plaquettes dans le sang (kiloplaquettes/mL) |
| `serum_creatinine` | Taux de créatinine sérique dans le sang |
| `serum_sodium` | Taux de sodium sérique dans le sang |
| `sex` | Sexe du patient (binaire) |
| `smoking` | Si le patient fume |
| `time` | Durée du suivi (jours) |
| `DEATH_EVENT` | **Cible** — si le patient est décédé (1) ou non (0) |

---

##  Structure du projet

```
HEART-FAILURE-PROJECT/
│
├── data/
│   └── heart_failure_clinical_records_dataset.csv
│
├── notebooks/
│   └── eda.ipynb                  # Analyse exploratoire des données
│
├── src/
│   ├── data_processing.py         # Chargement, nettoyage, optimisation mémoire
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   └── evaluate_model.py          # Comparaison des modèles et sélection du meilleur
│
├── app/
│   └── app.py                     # Interface web Streamlit
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── lightgbm.pkl
│
├── tests/
│   ├── test_data_processing.py
│   └── test_evaluate_model.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installer les dépendances


```
pip install -r requirements.txt
```

---

##  Étape 1 — Traitement des données

**Fichier :** `src/data_processing.py`

Ce module gère toutes les tâches de préparation des données :

- **Chargement des données** du fichier CSV brut
- **Vérification des valeurs manquantes** — aucune valeur manquante trouvée dans ce jeu de données
- **Détection des outliers** — en utilisant la méthode IQR
- **Gestion du déséquilibre de classes** — en utilisant la méthode class_weight="balanced" (67.9% survivants vs 32.1% décédés)
- **Optimisation de la mémoire** via la fonction `optimize_memory(df)` :

```python
def optimize_memory(df):
    """
    Réduit la mémoire utilisée par le DataFrame en convertissant
    les types de données vers des types moins lourds :
        - float64 (8 octets) → float32 (4 octets)
        - int64   (8 octets) → int32   (4 octets)
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df
```

L'amélioration de la mémoire est démontrée dans `notebooks/eda.ipynb`.

---


##  Étape 2 — Entraînement des modèles

**Fichiers :** `src/train_*.py`

Quatre modèles ont été entraînés et évalués :

| Modèle | Script |
|---|---|
| Logistic Regression | `train_logistic_regression.py` |
| Random Forest | `train_random_forest.py` |
| XGBoost | `train_xgboost.py` |
| LightGBM | `train_lightgbm.py` |

Pour entraîner tous les modèles :

```bash
cd src
python train_logistic_regression.py
python train_random_forest.py
python train_xgboost.py
python train_lightgbm.py
```

Chaque modèle entraîné est sauvegardé en fichier `.pkl` dans le dossier `models/`.

---

##  Étape 3 — Évaluation des modèles

**Fichier :** `src/evaluate_model.py`

Tous les modèles sont évalués selon les métriques suivantes :
- **ROC-AUC** (métrique principale — la plus critique en contexte médical)
- **Rappel** (prioritaire — minimiser les faux négatifs est vital)
- **F1-Score**
- **Accuracy**
- **Précision**

Pour évaluer et sélectionner le meilleur modèle :

```bash
python src/evaluate_model.py
```

---

##  Étape 4 — Explicabilité SHAP

SHAP (SHapley Additive exPlanations) est intégré pour rendre le modèle transparent :

- **Summary Plot** — importance globale des features sur tous les patients
- **Waterfall Plot** — explication au niveau d'un patient individuel (montre exactement quelles features ont augmenté ou diminué la prédiction)
- **Force Plot** — décomposition visuelle d'une prédiction unique

Résultat clé(corrélations entre features) : `time`, `ejection_fraction` et `serum_creatinine` sont les 3 features les plus influentes.

---

##  Étape 5 — Interface Web (Streamlit)

**Fichier :** `app/app.py`

Pour lancer l'application :

```bash
# S'assurer que le .venv est activé au préalable
streamlit run app/app.py
```

L'interface permet aux médecins de :
1. **Saisir les données cliniques du patient** via des curseurs et menus déroulants
2. **Visualiser la prédiction** (risque de décès : Faible / Élevé) avec la probabilité associée
3. **Explorer les explications SHAP** pour le patient spécifique

---

##  Étape 6 — Tests automatisés

**Fichiers :** `tests/`

```bash
pytest tests/
```

Tests inclus :

| Test | Description |
|---|---|
| `test_data_processing.py` | Vérifie le traitement des valeurs manquantes et la fonction `optimize_memory()` |
| `test_evaluate_model.py` | Vérifie le chargement du modèle et le format de sortie des prédictions |

---

##  Étape 7 — Pipeline CI/CD (GitHub Actions)

**Fichier :** `.github/workflows/ci.yml`

Le pipeline s'exécute automatiquement à chaque `push` ou `pull request` vers `main` :

1. Configuration de l'environnement Python
2. Installation de toutes les dépendances depuis `requirements.txt`
3. Exécution de tous les tests avec `pytest`

Cela garantit que le code est toujours dans un état fonctionnel.

---

##  Questions critiques 

### les données était-il équilibré ?
Non. Le jeu de données est déséquilibré (67.9% survivants, 32.1% décédés). Nous avons appliqué **class_weight="balanced"** .  réduisant significativement les patients à haut risque non détectés.

### Quel modèle ML a obtenu les meilleures performances ?

| Modèle | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | 0.8549 | 0.8000 | 0.7333 | 0.5789 | 0.6471 |
| Random Forest | 0.9050 | 0.8333 | 0.8000 | 0.6316 | 0.7059 |
| LightGBM | 0.8472 | 0.8333 | 0.8462 | 0.5789 | 0.6875 |
| XGBoost | 0.8678 | 0.8167 | 0.7500 | 0.6316 | 0.6857 |

**Random Forest** a été sélectionné comme meilleur modèle sur la base du score combiné entre AUC et Recall (0.5×AUC + 0.5×Recall)
    ROC-AUC      → Random Forest (0.9050)
    Accuracy     → Random Forest (0.8333)
    Recall       → Random Forest (0.6316)
    F1-Score     → Random Forest (0.7059)

### Quelles features médicales ont le plus influencé les prédictions (résultats SHAP) ?
1. `time` — Durée du suivi (plus longue = risque plus faible)
2. `ejection_fraction` — Faible fraction d'éjection = risque plus élevé
3. `serum_creatinine` — Taux élevés fortement associés à la mortalité

### Quels enseignements le prompt engineering a-t-il apporté ?
Voir la section dédiée ci-dessous.

---

## 💡 Documentation Prompt Engineering

**Tâche sélectionnée :** Fonction d'optimisation de la mémoire (`optimize_memory`)

**Prompt utilisé :**
> *"Écris une fonction Python appelée optimize_memory(df) qui réduise la mémoire en convertissant les colonnes float64 en float32 et les colonnes int64 en int32. en affichant l'utilisation mémoire avant et après."*

**Résultat :** La fonction générée était immédiatement utilisable et a démontré une réduction de 49.8% de l'utilisation mémoire sur ce jeu de données.

**Efficacité :** Le prompt a été très efficace car il était précis sur le nom de la fonction, les entrées/sorties et les conversions exactes nécessaires. L'exigence d'afficher les statistiques avant/après a rendu le résultat directement utilisable dans le notebook.


---

## Résumé pour faire fonctionner le projet
pip install -r requirements.txt

python src/train_lightgbm.py

python src/train_logistic_regression.py

python src/train_random_forest.py

python src/train_xgboost.py

streamlit run app/app.py