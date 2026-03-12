"""
app.py
======
Interface Streamlit — Prédiction du risque d'insuffisance cardiaque.

Lancer l'application :
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # évite les erreurs tkinter sur Windows
import joblib
import shap
import os

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Heart Failure Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def load_model():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path  = os.path.join(base_dir, 'models', 'best_model.pkl')

    if not os.path.exists(model_path):
        return None, None

    model_data = joblib.load(model_path)

    # Le modèle est sauvegardé en dict {model, scaler, features}
    if isinstance(model_data, dict):
        model  = model_data['model']
        scaler = model_data['scaler']
    else:
        model  = model_data
        scaler = None

    return model, scaler

# ============================================================
# EN-TÊTE
# ============================================================
st.title("🫀 Heart Failure Risk Prediction")
st.markdown("""
> **Outil d'aide à la décision clinique** — Prédit le risque de décès par insuffisance cardiaque
> en utilisant un modèle **Random Forest** (ROC-AUC = 0.905) avec explainabilité **SHAP**.
""")

# Vérification que le modèle existe
model, scaler = load_model()

if model is None:
    st.error("❌ Modèle introuvable ! Lance d'abord : `python src/train_random_forest.py`")
    st.stop()

st.success("✅ Modèle chargé — Random Forest (ROC-AUC = 0.905)")
st.divider()

# ============================================================
# FORMULAIRE PATIENT
# ============================================================
st.header("📋 Données du Patient")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Données Biologiques")

    age = st.slider(
        "🎂 Âge (années)", 40, 95, 60,
        help="Âge du patient"
    )

    ejection_fraction = st.slider(
        "💓 Fraction d'éjection (%)", 14, 80, 38,
        help="% de sang éjecté à chaque battement (normal: 55-70%)"
    )

    serum_creatinine = st.number_input(
        "🧪 Créatinine sérique (mg/dL)", 0.5, 10.0, 1.1, 0.1,
        help="Indicateur de la fonction rénale (normal: 0.6-1.2)"
    )

    serum_sodium = st.slider(
        "🧂 Sodium sérique (mEq/L)", 110, 150, 137,
        help="Niveau de sodium dans le sang (normal: 135-145)"
    )

    creatinine_phosphokinase = st.number_input(
        "⚗️ CPK (mcg/L)", 20, 8000, 250, 10,
        help="Enzyme créatinine phosphokinase (normal: 10-120)"
    )

    platelets = st.number_input(
        "🔬 Plaquettes (kiloplaquettes/mL)", 25000, 850000, 250000, 1000,
        help="Nombre de plaquettes (normal: 150000-400000)"
    )

with col2:
    st.subheader("Antécédents Médicaux")

    anaemia = st.radio(
        "🩸 Anémie",
        options=[0, 1],
        format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
        horizontal=True,
        help="Réduction des globules rouges ou hémoglobine"
    )

    diabetes = st.radio(
        "🍬 Diabète",
        options=[0, 1],
        format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
        horizontal=True
    )

    high_blood_pressure = st.radio(
        "🩺 Hypertension",
        options=[0, 1],
        format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
        horizontal=True
    )

    smoking = st.radio(
        "🚬 Fumeur",
        options=[0, 1],
        format_func=lambda x: "❌ Non" if x == 0 else "✅ Oui",
        horizontal=True
    )

    sex = st.radio(
        "👤 Sexe",
        options=[0, 1],
        format_func=lambda x: "Femme" if x == 0 else "Homme",
        horizontal=True
    )

    time = st.slider(
        "📅 Période de suivi (jours)", 4, 285, 100,
        help="Durée du suivi médical du patient"
    )

# ============================================================
# BOUTON DE PRÉDICTION
# ============================================================
st.divider()
predict_btn = st.button("🔍 Analyser le Risque du Patient", type="primary", use_container_width=True)

# ============================================================
# PRÉDICTION ET RÉSULTAT
# ============================================================
if predict_btn:

    # Créer le DataFrame patient dans le bon ordre
    patient_df = pd.DataFrame([{
        'age':                      age,
        'anaemia':                  anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes':                 diabetes,
        'ejection_fraction':        ejection_fraction,
        'high_blood_pressure':      high_blood_pressure,
        'platelets':                platelets,
        'serum_creatinine':         serum_creatinine,
        'serum_sodium':             serum_sodium,
        'sex':                      sex,
        'smoking':                  smoking,
        'time':                     time
    }])

    # Normaliser avec le scaler
    if scaler is not None:
        patient_scaled = scaler.transform(patient_df)
    else:
        patient_scaled = patient_df.values

    # Prédiction
    prediction  = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    risk_pct    = probability[1] * 100

    st.divider()
    st.header("📊 Résultat de l'Analyse")

    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if prediction == 1:
            st.error(f"""
            ## ⚠️ RISQUE ÉLEVÉ DE DÉCÈS
            ### Probabilité : **{risk_pct:.1f}%**

            Ce patient présente un **risque élevé** d'insuffisance cardiaque fatale.
            Une prise en charge médicale urgente est recommandée.
            """)
        else:
            st.success(f"""
            ## ✅ RISQUE FAIBLE
            ### Probabilité de décès : **{risk_pct:.1f}%**

            Ce patient présente un **faible risque** d'insuffisance cardiaque fatale.
            Un suivi régulier est recommandé.
            """)

        # Métriques rapides
        st.metric("Probabilité Survie",  f"{probability[0]*100:.1f}%")
        st.metric("Probabilité Décès",   f"{probability[1]*100:.1f}%")

    with col_gauge:
        # Graphique en barres horizontales
        fig, ax = plt.subplots(figsize=(6, 3))
        categories = ['Survie', 'Décès']
        values     = [probability[0]*100, probability[1]*100]
        colors     = ['#2ecc71', '#e74c3c']

        bars = ax.barh(categories, values, color=colors, edgecolor='white', height=0.5)
        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontweight='bold', fontsize=13)

        ax.set_xlim(0, 110)
        ax.set_xlabel('Probabilité (%)', fontsize=11)
        ax.set_title('Probabilités de Prédiction', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ============================================================
    # EXPLICATION SHAP
    # ============================================================
    st.divider()
    st.header("🔍 Explication SHAP — Pourquoi cette prédiction ?")

    col_info, _ = st.columns([2, 1])
    with col_info:
        st.info("""
        **Comment lire ce graphique :**
        - 🔴 **Barre vers la droite** → cette feature **augmente** le risque de décès
        - 🔵 **Barre vers la gauche** → cette feature **diminue** le risque de décès
        - **Plus la barre est longue**, plus l'impact est fort
        """)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_scaled)

        feature_names = list(patient_df.columns)

        # Récupérer les valeurs SHAP pour la classe 1 (décès)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Trier par importance absolue
        indices  = np.argsort(np.abs(sv))[::-1]
        features = [feature_names[i] for i in indices]
        values   = [sv[i] for i in indices]
        colors_s = ['#e74c3c' if v > 0 else '#3498db' for v in values]

        # Graphique SHAP
        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        bars = ax_shap.barh(features[::-1], values[::-1],
                            color=colors_s[::-1], edgecolor='white')
        ax_shap.axvline(x=0, color='black', linewidth=1)
        ax_shap.set_xlabel('Valeur SHAP (impact sur la prédiction)', fontsize=12)
        ax_shap.set_title('Impact de chaque Feature sur la Prédiction', fontsize=13, fontweight='bold')
        ax_shap.spines['top'].set_visible(False)
        ax_shap.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

        # Top 3 features
        st.subheader("🏆 Top 3 features les plus influentes :")
        cols_top = st.columns(3)
        for i, col in enumerate(cols_top):
            with col:
                direction = "⬆️ Augmente" if values[i] > 0 else "⬇️ Diminue"
                color_box = "🔴" if values[i] > 0 else "🔵"
                st.metric(
                    label=f"{color_box} {features[i]}",
                    value=f"{abs(values[i]):.4f}",
                    delta=direction
                )

    except Exception as e:
        st.warning(f"SHAP non disponible : {e}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🏥 Centrale Casablanca — Coding Week Mars 2026 | Heart Failure Prediction | Random Forest (AUC=0.905)")