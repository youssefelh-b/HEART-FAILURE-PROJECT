# ============================================================
# XGBOOST CLASSIFIER — INTÉGRATION AVEC data_processing.py
# ============================================================
# Tâches déjà gérées par data_processing.py (NE PAS REFAIRE) :
#   ✅ Chargement des données        → load_data()
#   ✅ Optimisation mémoire          → optimize_memory()
#   ✅ Séparation features / cible   → prepare_features()
#   ✅ Split Train / Test (80/20)    → split_data()
#   ✅ Normalisation StandardScaler  → normalize_data()
#
# Tâches gérées ICI uniquement :
#   ✅ Équilibrage SMOTE
#   ✅ Entraînement XGBoost
#   ✅ Prédictions + Métriques
#   ✅ Visualisations
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from data_processing import run_pipeline

import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('results', exist_ok=True)


# ============================================================
# 1. EXÉCUTION DU PIPELINE DE data_processing.py
# ============================================================
DATA_PATH = '../data/heart_failure_clinical_records_dataset.csv'

X_train_scaled, X_test_scaled, y_train, y_test, scaler = run_pipeline(DATA_PATH)


# ============================================================
# 2. ÉQUILIBRAGE AVEC SMOTE (sur X_train_scaled uniquement)
# ============================================================
print("\n" + "=" * 55)
print("ÉQUILIBRAGE AVEC SMOTE (train uniquement)")
print("=" * 55)
print(f"Avant SMOTE : {dict(y_train.value_counts())}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Après SMOTE : {dict(pd.Series(y_train_resampled).value_counts())}")
print(f"Taille train après SMOTE : {X_train_resampled.shape[0]} échantillons")
print("✅ Classes équilibrées !")


# ============================================================
# 3. ENTRAÎNEMENT DU MODÈLE XGBOOST
# ============================================================
print("\n" + "=" * 55)
print("ENTRAÎNEMENT XGBOOST")
print("=" * 55)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_resampled, y_train_resampled)
print("✅ Modèle XGBoost entraîné avec succès !")


# ============================================================
# 4. PRÉDICTIONS SUR LE JEU DE TEST
# ============================================================
y_pred       = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("✅ Prédictions générées !")


# ============================================================
# 5. MÉTRIQUES DE PERFORMANCE
# ============================================================
print("\n" + "=" * 55)
print("MÉTRIQUES DE PERFORMANCE")
print("=" * 55)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"\n📊 Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"\n📊 Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"\n📊 F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
print(f"\n📊 ROC-AUC   : {roc_auc:.4f}  ({roc_auc*100:.2f}%)")

print("\n" + "-" * 55)
print("RAPPORT COMPLET")
print("-" * 55)
print(classification_report(y_test, y_pred,
      target_names=['Survécu (0)', 'Décédé (1)']))


# ============================================================
# 6. CROSS-VALIDATION (5 folds)
# ============================================================
print("=" * 55)
print("CROSS-VALIDATION (5 folds)")
print("=" * 55)

cv_scores = cross_val_score(
    xgb_model,
    X_train_resampled,
    y_train_resampled,
    cv=5,
    scoring='roc_auc'
)

print(f"ROC-AUC par fold : {cv_scores.round(4)}")
print(f"Moyenne          : {cv_scores.mean():.4f}")
print(f"Écart-type       : {cv_scores.std():.4f}")

if cv_scores.std() < 0.05:
    print("✅ Modèle STABLE — généralise bien")
else:
    print("⚠️  Écart-type élevé — possible instabilité du modèle")


# ============================================================
# 7. VISUALISATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost — Résultats Complets', fontsize=16, fontweight='bold')

# --- Graphique 1 : Matrice de Confusion ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Survécu (0)', 'Décédé (1)'],
    yticklabels=['Survécu (0)', 'Décédé (1)'],
    ax=axes[0, 0]
)
axes[0, 0].set_title('Matrice de Confusion')
axes[0, 0].set_ylabel('Réel')
tn, fp, fn, tp = cm.ravel()
axes[0, 0].set_xlabel(f'Prédit  [TP={tp} | TN={tn} | FP={fp} | FN={fn}]')

# --- Graphique 2 : Courbe ROC ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='blue', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], color='gray',
                linestyle='--', label='Classifieur aléatoire')
axes[0, 1].set_xlabel('Taux Faux Positifs (FPR)')
axes[0, 1].set_ylabel('Taux Vrais Positifs (Recall)')
axes[0, 1].set_title('Courbe ROC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Graphique 3 : Comparaison des Métriques ---
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

bars = axes[1, 0].bar(metrics_names, metrics_values,
                      color=colors, edgecolor='black')
axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].set_title('Comparaison des Métriques')
axes[1, 0].set_ylabel('Score (0 à 1)')
axes[1, 0].axhline(y=0.8, color='red', linestyle='--',
                   alpha=0.5, label='Seuil 80%')
axes[1, 0].legend()

for bar, value in zip(bars, metrics_values):
    axes[1, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f'{value:.3f}',
        ha='center', fontweight='bold', fontsize=10
    )

# --- Graphique 4 : Importance des Features ---
feature_importance = pd.Series(
    xgb_model.feature_importances_,
    index=X_train_scaled.columns
).sort_values(ascending=True)

feature_importance.plot(
    kind='barh',
    ax=axes[1, 1],
    color='steelblue',
    edgecolor='black'
)
axes[1, 1].set_title("Importance des Features (XGBoost)")
axes[1, 1].set_xlabel("Score d'importance")

plt.tight_layout()
plt.savefig('results/xgboost_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Graphiques sauvegardés → results/xgboost_results.png")


# ============================================================
# 8. RÉSUMÉ FINAL
# ============================================================
print("\n" + "=" * 55)
print("RÉSUMÉ FINAL")
print("=" * 55)
results = {
    'Modèle'    : 'XGBoost + SMOTE',
    'Accuracy'  : f'{accuracy*100:.2f}%',
    'Precision' : f'{precision*100:.2f}%',
    'Recall'    : f'{recall*100:.2f}%',
    'F1-Score'  : f'{f1*100:.2f}%',
    'ROC-AUC'   : f'{roc_auc*100:.2f}%',
    'CV Mean'   : f'{cv_scores.mean()*100:.2f}%',
    'CV Std'    : f'{cv_scores.std()*100:.2f}%'
}
for key, value in results.items():
    print(f"  {key:<15} : {value}")

print("\n✅ Pipeline complet terminé avec succès !")