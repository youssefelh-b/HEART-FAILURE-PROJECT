"""
train_logistic.py
=================
Entraînement de la Régression Logistique.
Ce fichier fait UNIQUEMENT l'entraînement et la sauvegarde du modèle.
L'évaluation et la comparaison des modèles est faite dans evaluate_model.py.

Utilisation :
    python src/train_logistic.py
"""

import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import run_pipeline


# ============================================================
# 0. CHEMINS
# ============================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)  # crée le dossier models/ s'il n'existe pas


# ============================================================
# 1. PRÉPARER LES DONNÉES
# ============================================================

# run_pipeline() gère tout : chargement, optimisation mémoire,
# séparation features/cible, split train/test, normalisation.
# On récupère les données directement prêtes pour l'entraînement.

print("\n" + "="*50)
print("🚀 TRAIN — RÉGRESSION LOGISTIQUE")
print("="*50)

X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)


# ============================================================
# 2. CRÉER LE MODÈLE
# ============================================================

# max_iter=1000    : assez d'itérations pour que la descente de
#                    gradient converge et trouve les meilleurs poids
# class_weight=    : compense le déséquilibre 68% survivants / 32% décédés
#   'balanced'       sans ça le modèle ignorerait les patients à risque
# random_state=42  : résultats reproductibles à chaque exécution

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)


# ============================================================
# 3. ENTRAÎNER
# ============================================================

# .fit() lance la descente de gradient sur le train set.
# Le modèle ajuste les poids w₁, w₂, ... pour minimiser
# la log-loss jusqu'à convergence.

model.fit(X_train, y_train)
print("\n✅ Modèle logistic regression entraîné !")


# ============================================================
# 4. SAUVEGARDER
# ============================================================

# On sauvegarde 3 choses ensemble dans le même fichier :
#
#   model    : le modèle entraîné avec ses poids
#   scaler   : le StandardScaler appris sur le train set
#              (indispensable pour normaliser de nouvelles données
#               dans l'app Streamlit avec les mêmes paramètres)
#   features : les noms des colonnes dans le bon ordre
#              (pour éviter les erreurs de feature mismatch)

model_data = {
    'model'   : model,
    'scaler'  : scaler,
    'features': X_train.columns.tolist()
}

model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
joblib.dump(model_data, model_path)

print(f"✅ Modèle logistic regression sauvegardé → {model_path}")
print("\n➡️  Lance evaluate_model.py pour voir les performances !")