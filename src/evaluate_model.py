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
print("="*55)"""













































"""
evaluate_model.py
=================
Compare les 4 modèles entraînés et choisit le meilleur.

Métriques calculées pour chaque modèle :
    - ROC-AUC   : capacité à distinguer les classes
    - Accuracy  : taux de bonnes prédictions global
    - Precision : parmi les décédés prédits, combien sont vrais ?
    - Recall    : parmi les vrais décédés, combien détectés ?
    - F1-Score  : équilibre entre precision et recall

Usage:
    python src/evaluate_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import run_pipeline


# ============================================================
# CHEMINS
# ============================================================

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tous les modèles à comparer
MODELS = {
    'Logistic Regression': os.path.join(MODELS_DIR, 'logistic_regression.pkl'),
    'Random Forest':       os.path.join(MODELS_DIR, 'random_forest.pkl'),
    'LightGBM':            os.path.join(MODELS_DIR, 'lightgbm.pkl'),
    'XGBoost':             os.path.join(MODELS_DIR, 'xgboost.pkl'),
}


# ============================================================
# 1. PRÉPARER LES DONNÉES DE TEST
# ============================================================

def get_test_data():
    """
    Récupère les données de test via le pipeline.
    On utilise uniquement X_test et y_test pour l'évaluation.
    Le modèle n'a JAMAIS vu ces données pendant l'entraînement.
    """
    X_train, X_test, y_train, y_test, scaler = run_pipeline(DATA_PATH)
    return X_test, y_test


# ============================================================
# 2. CALCULER LES MÉTRIQUES D'UN MODÈLE
# ============================================================

def evaluate_one_model(model_name, model_path, X_test, y_test):
    """
    Charge un modèle et calcule toutes ses métriques sur le test set.

    Métriques expliquées :
        ROC-AUC   : 1.0 = parfait, 0.5 = aléatoire
        Accuracy  : % de bonnes prédictions toutes classes confondues
        Precision : sur les patients prédits décédés, combien le sont vraiment ?
        Recall    : sur les vrais décédés, combien a-t-on détectés ?
                    → LE PLUS IMPORTANT en médecine (minimiser les faux négatifs)
        F1-Score  : moyenne harmonique precision/recall

    Args:
        model_name  : nom du modèle (pour l'affichage)
        model_path  : chemin vers le fichier .pkl
        X_test      : features de test (déjà normalisées)
        y_test      : vraies étiquettes de test

    Returns:
        dict : toutes les métriques + proba pour la courbe ROC
    """
    # Vérifier que le fichier existe
    if not os.path.exists(model_path):
        print(f"  ⚠️  {model_name} : fichier introuvable → {model_path}")
        return None

    # Charger le modèle
    model_data = joblib.load(model_path)

    # Récupérer le modèle (format dict ou directement)
    if isinstance(model_data, dict):
        model = model_data['model']
    else:
        model = model_data

    # Prédictions
    y_pred      = model.predict(X_test)

    # Probabilités pour ROC-AUC
    # LightGBM natif retourne directement des probabilités
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X_test)[:, 1]
elif hasattr(model, 'predict'):
    y_proba = model.predict(X_test)
    # Si valeurs > 1, c'est pas des probabilités
    if y_proba.max() > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Calculer les métriques
    metrics = {
        'Model':     model_name,
        'ROC-AUC':   round(roc_auc_score(y_test, y_proba), 4),
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, y_pred, zero_division=0), 4),
        '_y_proba':  y_proba,   # utilisé pour la courbe ROC (non affiché)
        '_y_pred':   y_pred,    # utilisé pour la matrice de confusion
    }

    return metrics


# ============================================================
# 3. AFFICHER LE TABLEAU DE COMPARAISON
# ============================================================

def print_comparison_table(results):
    """
    Affiche un tableau propre avec toutes les métriques.
    Met en évidence le meilleur modèle pour chaque métrique.
    """
    # Créer le DataFrame sans les colonnes internes
    df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in results
    ])
    df = df.set_index('Model')

    print("\n" + "=" * 70)
    print("📊 TABLEAU DE COMPARAISON DES MODÈLES")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Meilleur modèle pour chaque métrique
    print("\n🏆 MEILLEUR PAR MÉTRIQUE :")
    for col in ['ROC-AUC', 'Accuracy', 'Recall', 'F1-Score']:
        best = df[col].idxmax()
        val  = df[col].max()
        print(f"   {col:<12} → {best} ({val:.4f})")

    return df


# ============================================================
# 4. GRAPHIQUES
# ============================================================

def plot_metrics_comparison(df, results):
    """
    Génère 3 graphiques :
        1. Barplot de toutes les métriques
        2. Courbes ROC
        3. Matrices de confusion
    """
    colors  = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    metrics = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    models  = df.index.tolist()

    # ── Graphique 1 : Barplot des métriques ─────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x      = np.arange(len(metrics))
    width  = 0.18

    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [df.loc[model_name, m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold'
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des Métriques — 4 Modèles ML', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Seuil 0.8')
    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, 'metrics_comparison.png')
    plt.savefig(path1, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Graphique sauvegardé → {path1}")

    # ── Graphique 2 : Courbes ROC ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aléatoire (AUC=0.5)')

    for result, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(result['_y_test'], result['_y_proba'])
        auc_val      = result['ROC-AUC']
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{result['Model']} (AUC={auc_val:.3f})")

    ax.set_xlabel('Taux Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux Vrais Positifs (TPR)', fontsize=12)
    ax.set_title('Courbes ROC — Comparaison des Modèles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.savefig(path2, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Courbes ROC sauvegardées → {path2}")

    # ── Graphique 3 : Matrices de confusion ─────────────────
    fig, axes = plt.subplots(1, len(results), figsize=(16, 4))

    for ax, result, color in zip(axes, results, colors):
        cm = confusion_matrix(result['_y_test'], result['_y_pred'])
        im = ax.imshow(cm, cmap='Blues')

        # Annoter les cellules
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Prédit Survivant', 'Prédit Décédé'], fontsize=8)
        ax.set_yticklabels(['Vrai Survivant', 'Vrai Décédé'], fontsize=8)
        ax.set_title(f"{result['Model']}\nRecall={result['Recall']:.2f}",
                     fontweight='bold', fontsize=10)

    plt.suptitle('Matrices de Confusion — 4 Modèles', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    plt.savefig(path3, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Matrices de confusion sauvegardées → {path3}")


# ============================================================
# 5. CHOISIR LE MEILLEUR MODÈLE
# ============================================================

def choose_best_model(df, results):
    """
    Choisit le meilleur modèle selon ROC-AUC et Recall combinés.

    Pourquoi Recall ?
        En médecine, rater un patient à risque (faux négatif) est
        BIEN PLUS GRAVE qu'une fausse alarme (faux positif).
        → On maximise donc le Recall en priorité.

    Score combiné = 0.5 × ROC-AUC + 0.5 × Recall
    """
    # Score combiné = 50% AUC + 50% Recall
    df['Combined Score'] = 0.5 * df['ROC-AUC'] + 0.5 * df['Recall']
    best_name = df['Combined Score'].idxmax()
    best_row  = df.loc[best_name]

    print("\n" + "=" * 70)
    print("🏆 MODÈLE SÉLECTIONNÉ")
    print("=" * 70)
    print(f"\n  ✅ {best_name}")
    print(f"\n  Justification :")
    print(f"     ROC-AUC   : {best_row['ROC-AUC']:.4f}")
    print(f"     Recall    : {best_row['Recall']:.4f}  ← prioritaire en médecine")
    print(f"     F1-Score  : {best_row['F1-Score']:.4f}")
    print(f"     Accuracy  : {best_row['Accuracy']:.4f}")
    print(f"\n  Combined Score (0.5×AUC + 0.5×Recall) : {best_row['Combined Score']:.4f}")
    print("\n" + "=" * 70)

    # Sauvegarder le meilleur modèle sous un nom générique
    # pour que app.py puisse le charger facilement
    model_file = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Random Forest':       'random_forest.pkl',
        'LightGBM':            'lightgbm.pkl',
        'XGBoost':             'xgboost.pkl',
    }[best_name]

    src  = os.path.join(MODELS_DIR, model_file)
    dst  = os.path.join(MODELS_DIR, 'best_model.pkl')
    data = joblib.load(src)
    joblib.dump(data, dst)
    print(f"  💾 Meilleur modèle sauvegardé → {dst}")
    print(f"     (utilisé par app.py et evaluate_model.py)")

    return best_name


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("📊 ÉVALUATION ET COMPARAISON DES 4 MODÈLES ML")
    print("=" * 70)

    # Préparer les données de test
    print("\n📂 Chargement des données de test...")
    X_test, y_test = get_test_data()

    # Évaluer chaque modèle
    print("\n🔍 Évaluation des modèles...")
    results = []

    for name, path in MODELS.items():
        print(f"\n  → {name}")
        metrics = evaluate_one_model(name, path, X_test, y_test)
        if metrics is not None:
            metrics['_y_test'] = y_test   # pour les courbes ROC
            results.append(metrics)

    if not results:
        print("❌ Aucun modèle trouvé — Lance d'abord les fichiers train_*.py")
        sys.exit(1)

    # Tableau de comparaison
    df = print_comparison_table(results)

    # Rapport détaillé par modèle
    print("\n" + "=" * 70)
    print("📋 RAPPORT DÉTAILLÉ PAR MODÈLE")
    print("=" * 70)
    for result in results:
        print(f"\n── {result['Model']} ──")
        print(classification_report(
            result['_y_test'], result['_y_pred'],
            target_names=['Survivant (0)', 'Décédé (1)']
        ))

    # Graphiques
    print("\n📈 Génération des graphiques...")
    plot_metrics_comparison(df, results)

    # Choisir le meilleur
    best = choose_best_model(df, results)

    print(f"\n🎯 Prochaine étape : streamlit run app/app.py")