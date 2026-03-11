"""
train_model.py
==============
Ce fichier entraîne un modèle Random Forest sur les données de prédiction
de l'insuffisance cardiaque.

Il s'appuie sur le pipeline de data_processing.py pour préparer les données,
puis entraîne et sauvegarde le modèle final.

Usage:
    python src/train_model.py
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_processing import run_pipeline


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 1. ENTRAÎNEMENT DU MODÈLE
# ============================================================

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Entraîne un Random Forest avec class_weight='balanced' pour
    compenser le déséquilibre des classes (68% survivants / 32% décédés).

    Hyperparamètres choisis :
        - n_estimators=200  : 200 arbres (bon compromis biais/variance)
        - max_depth=10      : limite l'overfitting
        - class_weight='balanced' : pénalise plus les erreurs sur la classe minoritaire
        - random_state=42   : reproductibilité

    Args:
        X_train : Features d'entraînement (normalisées)
        y_train : Target d'entraînement

    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    print("\n🌲 Entraînement du Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',   # gère le déséquilibre des classes
        random_state=42,
        n_jobs=-1                  # utilise tous les cœurs CPU
    )

    model.fit(X_train, y_train)

    print(f"✅ Modèle entraîné sur {X_train.shape[0]} patients")
    print(f"   Nombre d'arbres  : {model.n_estimators}")
    print(f"   Profondeur max   : {model.max_depth}")
    print(f"   class_weight     : {model.class_weight}")

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
        model  : Modèle RandomForest entraîné
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
        model  : RandomForestClassifier chargé
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
    print("🚀 ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
    print("=" * 50)


    # Étape 1 : Pipeline de données
    X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)

    # Étape 2 : Entraînement
    model = train_model(X_train, y_train)

    # Étape 3 : Sauvegarde
    save_model(model, scaler)

    print("\n" + "=" * 50)
    print("✅ Modèle Random Forest prêt — Lance evaluate_model.py pour les métriques !")
    print("=" * 50)