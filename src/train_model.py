"""
train_model.py
==============
Ce fichier contient la fonction d'entrainement des différents modèles.
Il reutilise directement la pipeline de data processing du projet.

Pipeline :
    run_pipeline() -> train_NAME_OF_MODEL() -> evaluate_model()
"""

import contextlib
import io
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from data_processing import run_pipeline

# ========================================================================================================================
# 1. MODELE LIGHTGBM
# ========================================================================================================================

# ============================================================
# 1.1 PARAMETRES LIGHTGBM
# ============================================================

LGB_PARAMS = {
    # Tache de classification binaire (0 = survivant, 1 = décédé)
    "objective": "binary",

    # Fonction de perte surveillée pendant l'entraînement
    "metric": "binary_logloss",

    # Algorithme de boosting : GBDT (Gradient Boosted Decision Trees)
    # Construit les arbres séquentiellement, chacun corrigeant les erreurs du précédent
    "boosting_type": "gbdt",

    # Nombre maximum de feuilles par arbre
    # 31 est la valeur par défaut LightGBM
    "num_leaves": 31,

    # Taux d'apprentissage : contrôle la correction apportée par chaque arbre
    "learning_rate": 0.05,

    # Fraction des features utilisées pour construire chaque arbre (régularisation)
    "feature_fraction": 0.9,

    # Graine aléatoire pour la reproductibilité des résultats
    "seed": 42,

    # Supprime tous les messages Info/Warning de LightGBM pendant l'entraînement
    "verbosity": -1,
}


# ============================================================
# 1.2 ENTRAINEMENT DU MODELE
# ============================================================

def train_lightgbm(data_path: str, num_boost_round: int = 100, threshold: float = 0.5):
    """
    Entraine un modele LightGBM a partir du CSV du projet.

    Etapes :
        1. Executer la pipeline de traitement des donnees
        2. Construire le Dataset LightGBM
        3. Entrainement du classifieur
        4. Calcul des metriques (ROC-AUC, Accuracy, Precision, Recall, F1-score)

    Args:
        data_path (str): Chemin vers le fichier CSV
        num_boost_round (int): Nombre d'iterations d'entrainement
        threshold (float): Seuil de classification pour convertir les probabilites en classes

    Returns:
        dict: Dictionnaire contenant le modele, les metriques et les predictions
    """
    # Etape 1 : Recuperer les donnees preprocesses (silencieux)
    with contextlib.redirect_stdout(io.StringIO()):
        X_train, X_test, y_train, y_test, _ = run_pipeline(data_path)

    # Etape 2 : Creer le dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)

    # Etape 3 : Entrainer le modele
    classifier = lgb.train(LGB_PARAMS, train_data, num_boost_round=num_boost_round)

    # Etape 4 : Predire et evaluer
    y_pred_proba = classifier.predict(X_test)
    y_pred = (y_pred_proba > threshold).astype(int)

    roc_auc_lightgbm = roc_auc_score(y_test, y_pred_proba)
    accuracy_lightgbm = accuracy_score(y_test, y_pred)
    precision_lightgbm = precision_score(y_test, y_pred, zero_division=0)
    recall_lightgbm = recall_score(y_test, y_pred, zero_division=0)
    f1_lightgbm = f1_score(y_test, y_pred, zero_division=0)

    return {
        "model": classifier,
        "roc_auc_lightgbm": roc_auc_lightgbm,
        "accuracy_lightgbm": accuracy_lightgbm,
        "precision_lightgbm": precision_lightgbm,
        "recall_lightgbm": recall_lightgbm,
        "f1_lightgbm": f1_lightgbm,
        "y_pred_proba_lightgbm": np.asarray(y_pred_proba),
        "y_pred_lightgbm": np.asarray(y_pred),
    }


# ============================================================
# 1.3 EXECUTION DIRECTE
# ============================================================

def main() -> None:
    """
    Point d'entree principal pour lancer l'entrainement en local.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "heart_failure_clinical_records_dataset.csv")

    lightgbm_results = train_lightgbm(data_path=data_path)

    print("\n" + "=" * 50)
    print("📊 RESULTATS LIGHTGBM")
    print("=" * 50)
    print(f"  ROC AUC   : {lightgbm_results['roc_auc_lightgbm']:.4f}")
    print(f"  Accuracy  : {lightgbm_results['accuracy_lightgbm']:.4f}")
    print(f"  Precision : {lightgbm_results['precision_lightgbm']:.4f}")
    print(f"  Recall    : {lightgbm_results['recall_lightgbm']:.4f}")
    print(f"  F1-score  : {lightgbm_results['f1_lightgbm']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
