"""
train_model.py
==============
Ce fichier entraîne un modèle LightGBM sur les données de prédiction
de l'insuffisance cardiaque.

Il s'appuie sur le pipeline de data_processing.py pour préparer les données,
puis entraîne et sauvegarde le modèle final.

Usage:
    python src/train_model.py
"""

import os
import joblib
import lightgbm as lgb
from data_processing import run_pipeline


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'lightgbm.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_lgb.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. ENTRAÎNEMENT DU MODÈLE
# ============================================================

LGB_PARAMS = {
    "objective": "binary",        # classification binaire (0 = survivant, 1 = décédé)
    "metric": "binary_logloss",   # fonction de perte surveillée pendant l'entraînement
    "boosting_type": "gbdt",      # Gradient Boosted Decision Trees
    "num_leaves": 31,             # nombre maximum de feuilles par arbre
    "learning_rate": 0.05,        # contrôle la correction apportée par chaque arbre
    "feature_fraction": 0.9,      # fraction des features par arbre (régularisation)
    "seed": 42,                   # reproductibilité
    "verbosity": -1,              # supprime les messages Info/Warning de LightGBM
}

NUM_BOOST_ROUND = 100


def train_model(X_train, y_train):
    """
    Entraîne un modèle LightGBM (GBDT) pour la classification binaire.

    Hyperparamètres choisis :
        - num_leaves=31       : taille des arbres (défaut LightGBM)
        - learning_rate=0.05  : pas d'apprentissage conservateur
        - feature_fraction=0.9: sous-échantillonnage des features (régularisation)
        - num_boost_round=100 : nombre d'itérations de boosting
        - seed=42             : reproductibilité

    Args:
        X_train : Features d'entraînement (normalisées)
        y_train : Target d'entraînement

    Returns:
        lgb.Booster: Modèle entraîné
    """
    print("\n⚡ Entraînement du LightGBM...")

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(LGB_PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)

    print(f"✅ Modèle entraîné sur {X_train.shape[0]} patients")
    print(f"   Nombre d'itérations : {NUM_BOOST_ROUND}")
    print(f"   Boosting type       : {LGB_PARAMS['boosting_type']}")
    print(f"   Learning rate       : {LGB_PARAMS['learning_rate']}")

    return model


# ============================================================
# 2. SAUVEGARDE DU MODÈLE
# ============================================================

def save_model(model, scaler):
    """
    Sauvegarde le modèle et le scaler sur disque avec joblib.

    Les deux doivent être sauvegardés ensemble car le scaler
    est nécessaire pour transformer les nouvelles données en production.

    Args:
        model  : Modèle LightGBM entraîné
        scaler : StandardScaler ajusté sur X_train
    """
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n💾 Modèle sauvegardé  → {MODEL_PATH}")
    print(f"💾 Scaler sauvegardé  → {SCALER_PATH}")


# ============================================================
# 3. CHARGEMENT DU MODÈLE
# ============================================================

def load_model():
    """
    Charge le modèle et le scaler depuis le disque.

    Returns:
        model  : lgb.Booster chargé
        scaler : StandardScaler chargé

    Raises:
        FileNotFoundError: si les fichiers n'existent pas encore
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord : python src/train_model.py"
        )

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"✅ Modèle chargé  ← {MODEL_PATH}")
    print(f"✅ Scaler chargé  ← {SCALER_PATH}")

    return model, scaler


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 ENTRAÎNEMENT DU MODÈLE LIGHTGBM")
    print("=" * 50)

    # Étape 1 : Pipeline de données
    X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)

    # Étape 2 : Entraînement
    model = train_model(X_train, y_train)

    # Étape 3 : Sauvegarde
    save_model(model, scaler)

    print("\n" + "=" * 50)
    print("✅ Modèle LightGBM prêt — Lance evaluate_model.py pour les métriques !")
    print("=" * 50)
