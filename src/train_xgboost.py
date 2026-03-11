"""
train_xgboost.py
================
Entraînement du modèle XGBoost.
Ce fichier fait UNIQUEMENT l'entraînement et la sauvegarde du modèle.
L'évaluation et la comparaison des modèles est faite dans evaluate_model.py.

Utilisation :
    python src/train_xgboost.py
"""

import os
import sys
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import run_pipeline


# ============================================================
# 0. CHEMINS
# ============================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 1. PRÉPARER LES DONNÉES
# ============================================================

print("\n" + "="*50)
print("🚀 TRAIN — XGBOOST")
print("="*50)

X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)


# ============================================================
# 2. SMOTE — uniquement sur le train set
# ============================================================

# XGBoost ne supporte pas class_weight='balanced' comme sklearn.
# On utilise donc SMOTE pour rééquilibrer les classes.
# SMOTE génère des exemples synthétiques de la classe minoritaire (décès=1)
# par interpolation entre des vrais exemples existants.
# ⚠️ SMOTE s'applique UNIQUEMENT sur le train set, jamais sur le test set.

print(f"\n  Avant SMOTE → Classe 0: {(y_train==0).sum()} | Classe 1: {(y_train==1).sum()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"  Après SMOTE → Classe 0: {(y_train_resampled==0).sum()} | Classe 1: {(y_train_resampled==1).sum()}")
print("✅ Classes équilibrées !")


# ============================================================
# 3. CRÉER LE MODÈLE
# ============================================================

# n_estimators=100  : nombre d'arbres construits séquentiellement
# max_depth=4       : profondeur max de chaque arbre, limite l'overfitting
# learning_rate=0.1 : vitesse d'apprentissage
# subsample=0.8     : utilise 80% des données à chaque arbre (robustesse)
# colsample_bytree  : utilise 80% des features à chaque arbre (robustesse)
# eval_metric       : évite un warning de la librairie XGBoost

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)


# ============================================================
# 4. ENTRAÎNER
# ============================================================

model.fit(X_train_resampled, y_train_resampled)
print("\n✅ Modèle xgboost entraîné !")


# ============================================================
# 5. SAUVEGARDER
# ============================================================

model_data = {
    'model'   : model,
    'scaler'  : scaler,
    'features': X_train.columns.tolist()
}

model_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
joblib.dump(model_data, model_path)

print(f"✅ Modèle xgboost sauvegardé → {model_path}")
print("\n➡️  Lance evaluate_model.py pour voir les performances !")