"""
evaluate_model.py
=================
Compare les 4 modèles entraînés :
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM

Ce fichier charge les modèles sauvegardés (.pkl), calcule toutes
les métriques, génère les visualisations et désigne le meilleur modèle.

⚠️  Lancer d'abord les 4 fichiers d'entraînement avant ce fichier :
    python src/train_logistic_regression.py
    python src/train_model_RandomForest.py
    python src/train_model_XGBoost.py
    python src/train_model_LightGBM.py

Utilisation :
    python src/evaluate_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import run_pipeline


# ============================================================
# 0. CHEMINS
# ============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. PRÉPARER LES DONNÉES
# ============================================================

# On recharge les données avec le même pipeline pour avoir
# X_test et y_test — les données que les modèles n'ont jamais vues.
# random_state=42 garantit exactement le même split qu'à l'entraînement.

print("\n" + "="*55)
print("📊 ÉVALUATION ET COMPARAISON DES MODÈLES")
print("="*55)

X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)


# ============================================================
# 2. CHARGER LES 4 MODÈLES
# ============================================================

# Dictionnaire : nom affiché → fichier .pkl
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest'      : 'random_forest.pkl',
    'XGBoost'            : 'xgboost.pkl',
    'LightGBM'           : 'lightgbm.pkl'
}

models = {}
for name, filename in MODEL_FILES.items():
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        loaded = joblib.load(path)
        models[name] = loaded['model'] if isinstance(loaded, dict) else loaded
        print(f"✅ {name} chargé")
    else:
        print(f"⚠️  {name} introuvable → lance d'abord son fichier d'entraînement")

if not models:
    print("\n❌ Aucun modèle trouvé. Lance d'abord les fichiers d'entraînement.")
    sys.exit(1)


# ============================================================
# 3. CALCULER LES MÉTRIQUES POUR CHAQUE MODÈLE
# ============================================================

print("\n" + "="*55)
print("📈 MÉTRIQUES DE PERFORMANCE")
print("="*55)

results = []

for name, model in models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        'Modèle'    : name,
        'Accuracy'  : accuracy_score(y_test, y_pred),
        'Precision' : precision_score(y_test, y_pred),
        'Recall'    : recall_score(y_test, y_pred),
        'F1-Score'  : f1_score(y_test, y_pred),
        'ROC-AUC'   : roc_auc_score(y_test, y_proba)
    })

# Tableau de comparaison
df_results     = pd.DataFrame(results).set_index('Modèle')
df_results_pct = (df_results * 100).round(2).astype(str) + '%'

print("\n" + df_results_pct.to_string())


# ============================================================
# 4. DÉSIGNER LE MEILLEUR MODÈLE
# ============================================================

# On utilise le ROC-AUC comme critère principal car :
# - Plus robuste que l'accuracy sur un dataset déséquilibré (68/32)
# - Mesure la capacité globale à distinguer les deux classes
# - Standard dans la littérature médicale pour les modèles de risque

best_model_name = df_results['ROC-AUC'].idxmax()
best_roc_auc    = df_results['ROC-AUC'].max()
best_model      = models[best_model_name]

print(f"\n🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"   ROC-AUC : {best_roc_auc:.4f}")
print(f"\n   Justification : ROC-AUC choisi comme critère principal")
print(f"   car plus fiable que l'accuracy sur un dataset déséquilibré (68/32).")
print(f"   Un ROC-AUC élevé garantit que le modèle distingue bien")
print(f"   les patients à risque des patients sains.")

# Rapport détaillé du meilleur modèle
y_pred_best = best_model.predict(X_test)
print(f"\n📋 Rapport détaillé — {best_model_name} :")
print(classification_report(y_test, y_pred_best,
      target_names=['Survie (0)', 'Décès (1)']))


# ============================================================
# 5. VISUALISATIONS
# ============================================================

n_models = len(models)
colors   = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Comparaison des Modèles — Heart Failure Prediction',
             fontsize=16, fontweight='bold')

# ── Graphique 1 : Comparaison des métriques (barres groupées) ──
ax1 = fig.add_subplot(3, 2, 1)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x     = np.arange(len(metrics_to_plot))
width = 0.8 / n_models

for i, (name, row) in enumerate(df_results.iterrows()):
    values = [row[m] for m in metrics_to_plot]
    ax1.bar(x + i * width, values, width, label=name,
            color=colors[i % len(colors)], edgecolor='black', alpha=0.85)

ax1.set_xticks(x + width * (n_models - 1) / 2)
ax1.set_xticklabels(metrics_to_plot)
ax1.set_ylim(0, 1.2)
ax1.set_title('Comparaison des Métriques', fontweight='bold')
ax1.legend(fontsize=8)
ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.4)

# ── Graphique 2 : Courbes ROC superposées ──────────────────────
ax2 = fig.add_subplot(3, 2, 2)
for i, (name, model) in enumerate(models.items()):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f'{name} (AUC={auc:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC=0.5)')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Courbes ROC — Tous les Modèles', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Graphiques 3-6 : Matrices de confusion pour chaque modèle ──
positions = [3, 4, 5, 6]
for pos, (name, model) in zip(positions, models.items()):
    ax = fig.add_subplot(3, 2, pos)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Survie', 'Décès'],
                yticklabels=['Survie', 'Décès'],
                cbar=False)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f'{name}\nTP={tp} | TN={tn} | FP={fp} | FN={fn}', fontsize=9)
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'models_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Graphiques sauvegardés → {plot_path}")
plt.show()


# ============================================================
# 6. SHAP — MEILLEUR MODÈLE UNIQUEMENT
# ============================================================

print("\n" + "="*55)
print(f"🔍 SHAP — {best_model_name}")
print("="*55)

# On génère les explications SHAP uniquement pour le meilleur modèle.
# LinearExplainer pour la régression logistique (modèle linéaire).
# TreeExplainer pour les modèles basés sur des arbres (RF, XGBoost, LightGBM).

from sklearn.linear_model import LogisticRegression as LR

if isinstance(best_model, LR):
    explainer   = shap.LinearExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    sv_plot     = shap_values
    base_val    = explainer.expected_value
else:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        sv_plot  = shap_values[1]
        base_val = explainer.expected_value[1]
    else:
        sv_plot  = shap_values
        base_val = explainer.expected_value

# Summary plot — quelles features influencent le plus le modèle ?
print(f"\n📊 SHAP Summary Plot — {best_model_name} :")
shap.summary_plot(sv_plot, X_test, show=True)

# Waterfall plot — explication pour le patient le plus à risque
most_at_risk = np.argmax(best_model.predict_proba(X_test)[:, 1])
proba        = best_model.predict_proba(X_test)[most_at_risk, 1]
realite      = 'Décédé' if y_test.values[most_at_risk] == 1 else 'Survivant'

print(f"\n🏥 Patient le plus à risque (index {most_at_risk}) :")
print(f"   Probabilité de décès : {proba*100:.1f}%")
print(f"   Réalité              : {realite}")

shap.waterfall_plot(shap.Explanation(
    values       = sv_plot[most_at_risk],
    base_values  = base_val,
    data         = X_test.iloc[most_at_risk],
    feature_names= X_test.columns.tolist()
))


# ============================================================
# 7. RÉSUMÉ FINAL
# ============================================================

print("\n" + "="*55)
print("🏆 RÉSUMÉ FINAL")
print("="*55)
print(df_results_pct.to_string())
print(f"\n✅ Meilleur modèle : {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")
print(f"   Ce modèle sera utilisé dans l'application Streamlit.")
print("="*55)
