# 🫀 HeartGuard — Prédiction d'Insuffisance Cardiaque

> **Medical Decision Support Application** — Predicting Heart Failure Risk with Explainable ML (SHAP)
> Coding Week · École Centrale Casablanca · Mars 2026

---

## 👥 Équipe

Groupe 31:
 Hatim EL GAOUTI,
 Adam SABILI,
 Youssef ELHALLAM,
 Mohamed EL YAAGOUBI,
 Ilyas LESSIQ

---

## 📋 Table des Matières

1. [Objectif du projet](#objectif-du-projet)
2. [Jeu de données](#jeu-de-données)
3. [Structure du projet](#structure-du-projet)
4. [Installation & Lancement](#installation--lancement)
5. [Pipeline ML](#pipeline-ml)
6. [Explicabilité SHAP](#explicabilité-shap)
7. [Interface Web](#interface-web-streamlit)
8. [Tests automatisés](#tests-automatisés)
9. [CI/CD GitHub Actions](#cicd-github-actions)
10. [Questions critiques](#questions-critiques)
11. [Prompt Engineering](#prompt-engineering)

---

## Objectif du projet

Outil avancé d'aide à la décision clinique conçu pour aider les médecins à prédire le risque de mortalité par insuffisance cardiaque, en s'appuyant sur des données cliniques réelles et un modèle de Machine Learning explicable via SHAP.

---

## Jeu de données

**Source :** [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords)

| Feature | Description |
|---|---|
| `age` | Âge du patient |
| `anaemia` | Diminution des globules rouges |
| `creatinine_phosphokinase` | Taux de l'enzyme CPK dans le sang |
| `diabetes` | Patient diabétique |
| `ejection_fraction` | % de sang éjecté à chaque battement |
| `high_blood_pressure` | Hypertension artérielle |
| `platelets` | Plaquettes dans le sang (kiloplaquettes/mL) |
| `serum_creatinine` | Créatinine sérique dans le sang |
| `serum_sodium` | Sodium sérique dans le sang |
| `sex` | Sexe du patient |
| `smoking` | Patient fumeur |
| `time` | Durée du suivi (jours) |
| `DEATH_EVENT` | **Cible** — décédé (1) ou non (0) |

---

## Structure du projet

```
HEART-FAILURE-PROJECT/
│
├── .github/workflows/
│   └── ci.yml                     # Pipeline CI/CD GitHub Actions
│
├── .streamlit/
│   └── config.toml                # Thème Streamlit (couleurs ECC)
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
│   └── evaluate_model.py          # Comparaison des modèles, sélection du meilleur
│
├── app/
│   └── app.py                     # Interface web Streamlit
│
├── models/                        # Modèles entraînés (.pkl) — générés après entraînement
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

## Installation & Lancement

### 1. Cloner le repo

```bash
git clone https://github.com/youssefelh-b/HEART-FAILURE-PROJECT.git
cd HEART-FAILURE-PROJECT
```

### 2. Créer et activer l'environnement virtuel

```bash
# Créer le venv
python -m venv .venv

# Activer — Windows
.venv\Scripts\activate

# Activer — Mac/Linux
source .venv/bin/activate
```

> ⚠️ **Important** : à chaque nouveau terminal, il faut réactiver le venv avant de lancer quoi que ce soit.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Entraîner les modèles

```bash
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_xgboost.py
python src/train_lightgbm.py
```

### 5. Évaluer et sélectionner le meilleur modèle

```bash
python src/evaluate_model.py
```

### 6. Lancer l'application

```bash
streamlit run app/app.py
```

---

## Pipeline ML

### Étape 1 — Traitement des données (`src/data_processing.py`)

- **Chargement** du fichier CSV
- **Valeurs manquantes** — aucune dans ce dataset
- **Outliers** — détection par méthode IQR
- **Déséquilibre de classes** — géré via `class_weight="balanced"` (67.9% survivants / 32.1% décédés)
- **Optimisation mémoire** via `optimize_memory(df)` :

```python
def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df
```

Réduction mémoire démontrée dans `notebooks/eda.ipynb` : **−49.8%**.

### Étape 2 — Modèles entraînés (`src/train_*.py`)

| Modèle | Script |
|---|---|
| Logistic Regression | `train_logistic_regression.py` |
| Random Forest | `train_random_forest.py` |
| XGBoost | `train_xgboost.py` |
| LightGBM | `train_lightgbm.py` |

Chaque modèle est sauvegardé en `.pkl` dans `models/`.

### Étape 3 — Évaluation (`src/evaluate_model.py`)

Métriques utilisées : **ROC-AUC** (principale), Recall, F1-Score, Accuracy, Précision.

| Modèle | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | 0.8549 | 0.8000 | 0.7333 | 0.5789 | 0.6471 |
| **Random Forest** | **0.9050** | **0.8333** | **0.8000** | **0.6316** | **0.7059** |
| LightGBM | 0.8472 | 0.8333 | 0.8462 | 0.5789 | 0.6875 |
| XGBoost | 0.8678 | 0.8167 | 0.7500 | 0.6316 | 0.6857 |

**→ Random Forest sélectionné** sur la base du score combiné `0.5×AUC + 0.5×Recall`.

---

## Explicabilité SHAP

SHAP (SHapley Additive exPlanations) est intégré pour rendre le modèle transparent :

- **Summary Plot** — importance globale des features sur l'ensemble du dataset
- **Waterfall Plot** — explication individuelle par patient
- **Force Plot** — décomposition visuelle d'une prédiction unique

**Top 3 features les plus influentes :**
1. `ejection_fraction` — faible fraction = risque élevé
2. `serum_creatinine` — taux élevés fortement associés à la mortalité
3. `time` — feature de suivi (à interpréter avec prudence : data leakage potentiel)

---

## Interface Web (Streamlit)

**Fichier :** `app/app.py`

L'interface permet aux médecins de :
1. **Saisir les données cliniques** via des sliders et menus
2. **Visualiser la prédiction** (risque Faible / Élevé) avec probabilité
3. **Explorer les explications SHAP** pour chaque patient

```bash
streamlit run app/app.py
```

---

## Tests automatisés

```bash
pytest tests/
```

| Fichier | Tests |
|---|---|
| `test_data_processing.py` | Valeurs manquantes, fonction `optimize_memory()` |
| `test_evaluate_model.py` | Chargement du modèle, format des prédictions |

---

## CI/CD GitHub Actions

**Fichier :** `.github/workflows/ci.yml`

Déclenché automatiquement à chaque `push` ou `pull request` sur `main` :

1. Configuration de l'environnement Python
2. Installation des dépendances depuis `requirements.txt`
3. Exécution de tous les tests avec `pytest`

---

## Questions critiques

### Le dataset était-il équilibré ?
Non — 67.9% survivants / 32.1% décédés. Géré via `class_weight="balanced"`, ce qui réduit significativement les faux négatifs (patients à risque non détectés).

### Quel modèle a obtenu les meilleures performances ?
**Random Forest** avec ROC-AUC = **0.905**. Voir tableau complet dans la section [Évaluation](#étape-3--évaluation-evaluatemodelpy).

### Quelles features ont le plus influencé les prédictions ?
`ejection_fraction`, `serum_creatinine` et `time` — voir section [SHAP](#explicabilité-shap).

### Quels enseignements le prompt engineering a-t-il apportés ?
Voir section suivante.

---

## Prompt Engineering

**Tâche sélectionnée :** Fonction `optimize_memory(df)`

**Prompt utilisé :**
> *"Écris une fonction Python appelée `optimize_memory(df)` qui réduit la mémoire en convertissant les colonnes `float64` en `float32` et les colonnes `int64` en `int32`, en affichant l'utilisation mémoire avant et après."*

**Résultat :** Fonction immédiatement utilisable, réduction mémoire de **49.8%** démontrée dans le notebook.

**Analyse :** Le prompt était efficace car il spécifiait le nom exact de la fonction, les types de conversions attendus, et le comportement de logging. La précision des instructions a éliminé les allers-retours et produit un code directement intégrable.

---

*École Centrale Casablanca · Coding Week · Mars 2026*