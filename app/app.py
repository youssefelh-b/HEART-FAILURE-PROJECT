"""
app.py
======
Interface Streamlit — Prédiction du risque d'insuffisance cardiaque.
Design : Centrale Casablanca — Coding Week Mars 2026

Lancer l'application :
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import shap
import os

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="HeartGuard — ECC Clinical AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS — DARK NAVY/TEAL THEME ECC
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600;700;800&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --ecc-navy:       #0D1F3C;
    --ecc-navy-mid:   #112347;
    --ecc-navy-card:  #162B52;
    --ecc-navy-light: #1E3A6E;
    --ecc-teal:       #00C2C2;
    --ecc-teal-dim:   #008F8F;
    --ecc-yellow:     #F5C842;
    --ecc-yellow-dim: #C9A030;
    --text-primary:   #E8EEF8;
    --text-body:      #B8C8E0;
    --text-muted:     #6A84A8;
    --border:         rgba(0,194,194,0.15);
    --border-soft:    rgba(255,255,255,0.06);
    --success:        #2DD4A0;
    --danger:         #F56565;
    --shadow:         0 8px 32px rgba(0,0,0,0.35);
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--ecc-navy) !important;
    color: var(--text-body) !important;
}

.stApp {
    background: linear-gradient(160deg, #0D1F3C 0%, #0A1828 50%, #0D2240 100%);
}

/* ── HEADER ── */
.ecc-header {
    background: linear-gradient(135deg, #0A1828 0%, var(--ecc-navy-mid) 50%, var(--ecc-navy-light) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 36px 44px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow), inset 0 1px 0 rgba(0,194,194,0.1);
}

.ecc-header::before {
    content: '';
    position: absolute;
    bottom: -60px;
    right: -60px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, var(--ecc-teal) 0%, transparent 70%);
    opacity: 0.07;
    border-radius: 50%;
}

.ecc-header::after {
    content: '';
    position: absolute;
    top: -40px;
    right: 100px;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, var(--ecc-yellow) 0%, transparent 70%);
    opacity: 0.06;
    border-radius: 50%;
}

.ecc-header-top {
    display: flex;
    align-items: center;
    gap: 28px;
    margin-bottom: 20px;
}

.ecc-logo {
    width: 72px;
    height: 72px;
    object-fit: contain;
    filter: brightness(0) invert(1);
    flex-shrink: 0;
    opacity: 0.92;
}

.ecc-header h1 {
    font-family: 'Raleway', sans-serif !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 0 5px 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
}

.ecc-teal-accent { color: var(--ecc-teal); }

.ecc-header .subtitle {
    color: var(--text-body);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    opacity: 0.85;
}

.ecc-badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.ecc-badge-yellow {
    background: var(--ecc-yellow);
    color: var(--ecc-navy);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    font-family: 'Raleway', sans-serif;
}

.ecc-badge-teal {
    background: rgba(0,194,194,0.15);
    border: 1px solid rgba(0,194,194,0.4);
    color: var(--ecc-teal);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* ── TEAM BAR ── */
.team-bar {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.team-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--ecc-teal-dim);
    font-weight: 700;
    font-family: 'Raleway', sans-serif;
    margin-right: 4px;
    white-space: nowrap;
}

.team-member {
    background: rgba(0,194,194,0.08);
    border: 1px solid rgba(0,194,194,0.2);
    color: var(--text-body);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    white-space: nowrap;
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Raleway', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    border-left: 4px solid var(--ecc-teal);
    padding-left: 14px;
    margin: 28px 0 18px 0;
}

/* ── MODEL STATUS ── */
.model-status {
    background: rgba(0,194,194,0.06);
    border: 1px solid rgba(0,194,194,0.25);
    border-left: 3px solid var(--ecc-teal);
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 0.875rem;
    color: var(--ecc-teal);
    font-weight: 600;
    margin-bottom: 20px;
}

/* ── RESULT CARDS ── */
.result-danger {
    background: linear-gradient(135deg, rgba(245,101,101,0.08), rgba(245,101,101,0.03));
    border: 1px solid rgba(245,101,101,0.3);
    border-left: 5px solid var(--danger);
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(245,101,101,0.1);
}

.result-safe {
    background: linear-gradient(135deg, rgba(45,212,160,0.08), rgba(45,212,160,0.03));
    border: 1px solid rgba(45,212,160,0.3);
    border-left: 5px solid var(--success);
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(45,212,160,0.1);
}

.result-number {
    font-family: 'Raleway', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 10px 0 6px 0;
}

.result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    font-weight: 700;
    font-family: 'Raleway', sans-serif;
}

.result-desc {
    font-size: 0.9rem;
    margin-top: 12px;
    color: var(--text-body);
    line-height: 1.55;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--ecc-teal-dim) 0%, var(--ecc-teal) 100%) !important;
    color: var(--ecc-navy) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Raleway', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 14px 28px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 6px 24px rgba(0,194,194,0.25) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--ecc-yellow-dim) 0%, var(--ecc-yellow) 100%) !important;
    box-shadow: 0 10px 32px rgba(245,200,66,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── METRICS ── */
[data-testid="stMetricValue"] {
    font-family: 'Raleway', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 700 !important;
}

/* ── SHAP INFO ── */
.shap-info {
    background: rgba(0,194,194,0.05);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 20px;
    line-height: 1.6;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border-soft) !important;
    margin: 24px 0 !important;
}

/* ── FOOTER ── */
.ecc-footer {
    text-align: center;
    padding: 28px 24px;
    color: var(--text-muted);
    font-size: 0.8rem;
    letter-spacing: 0.4px;
    border-top: 1px solid var(--border-soft);
    margin-top: 48px;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
}

.ecc-footer .highlight {
    color: var(--text-primary);
    font-weight: 700;
    font-family: 'Raleway', sans-serif;
}

.ecc-footer .teal { color: var(--ecc-teal); }
.ecc-footer .yellow { color: var(--ecc-yellow); }

.footer-team {
    margin-top: 12px;
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}

.footer-member {
    background: rgba(0,194,194,0.08);
    border: 1px solid rgba(0,194,194,0.18);
    color: var(--text-body);
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
}

/* ── FORM ── */
label {
    color: var(--text-body) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

.stNumberInput input {
    background: var(--ecc-navy-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.form-group-label {
    font-family: 'Raleway', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--ecc-teal);
    margin-bottom: 16px;
}

/* ── SLIDER — thumb + filled track + tick ── */
div[data-baseweb="slider"] [role="slider"] {
    background-color: var(--ecc-yellow) !important;
    border-color: var(--ecc-yellow) !important;
}

/* Filled track (the colored portion left of thumb) */
div[data-baseweb="slider"] div[data-testid="stSlider"] div,
div[data-baseweb="slider"] div[class*="track"] div,
div[data-baseweb="slider"] div[class*="Track"] div {
    background: var(--ecc-yellow) !important;
}

/* Brute-force: any div inside slider that has inline background set to red */
div[data-baseweb="slider"] div[style*="background-color: rgb(255"] {
    background-color: var(--ecc-yellow) !important;
}

div[data-baseweb="slider"] div[style*="background: rgb(255"] {
    background: var(--ecc-yellow) !important;
}

/* Streamlit injects inline styles — override with attribute selector */
[data-testid="stSlider"] div[style*="background-color"] {
    background-color: var(--ecc-yellow) !important;
}

/* Radio buttons */
[data-baseweb="radio"] div[style*="background-color: rgb(255"],
[data-baseweb="radio"] div[style*="border-color: rgb(255"],
[data-baseweb="radio"] div[style*="background: rgb(255"] {
    background-color: var(--ecc-yellow) !important;
    border-color: var(--ecc-yellow) !important;
}

/* Streamlit primary color CSS variable */
:root {
    --primary: #F5C842 !important;
    --primary-color: #F5C842 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def load_model():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path  = os.path.join(base_dir, 'models', 'best_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    if not os.path.exists(model_path):
        return None, None

    model_data = joblib.load(model_path)

    if isinstance(model_data, dict):
        model  = model_data['model']
        scaler = model_data['scaler']
    else:
        model  = model_data
        scaler = joblib.load(scaler_path)

    return model, scaler


model, scaler = load_model()


# ============================================================
# HEADER AVEC LOGO + ÉQUIPE
# ============================================================
st.markdown("""
<div class="ecc-header">
    <div class="ecc-header-top">
        <img class="ecc-logo"
             src="https://centrale-casablanca.ma/wp-content/uploads/2023/01/Logo.png"
             alt="ECC Logo" />
        <div>
            <h1>Heart<span class="ecc-teal-accent">Guard</span> — Clinical AI</h1>
            <p class="subtitle">Outil d'aide à la décision pour la prédiction du risque d'insuffisance cardiaque</p>
            <div class="ecc-badge-row">
                <span class="ecc-badge-yellow">🎓 Centrale Casablanca</span>
                <span class="ecc-badge-teal">Coding Week · Mars 2026</span>
            </div>
        </div>
    </div>
    <div class="team-bar">
        <span class="team-label">Team 31 —</span>
        <span class="team-member">Hatim EL GAOUTI</span>
        <span class="team-member">Adam SABILI</span>
        <span class="team-member">Youssef ELHALLAM</span>
        <span class="team-member">Mohamed EL YAAGOUBI</span>
        <span class="team-member">Ilyas LESSIQ</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Modèle introuvable — Lance d'abord : `python src/train_random_forest.py`")
    st.stop()

st.markdown('<div class="model-status">✅ Modèle chargé — Random Forest · ROC-AUC = 0.905</div>', unsafe_allow_html=True)


# ============================================================
# FORMULAIRE
# ============================================================
st.markdown('<div class="section-title">Données du Patient</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="form-group-label">🔬 Paramètres Biologiques</div>', unsafe_allow_html=True)

    age = st.slider("Âge (années)", 40, 95, 60,
        help="Âge du patient en années")

    ejection_fraction = st.slider("Fraction d'éjection (%)", 14, 80, 38,
        help="Pourcentage de sang éjecté à chaque battement · Normal : 55–70%")

    serum_creatinine = st.number_input("Créatinine sérique (mg/dL)", 0.5, 10.0, 1.1, 0.1,
        help="Indicateur de la fonction rénale · Normal : 0.6–1.2")

    serum_sodium = st.slider("Sodium sérique (mEq/L)", 110, 150, 137,
        help="Niveau de sodium dans le sang · Normal : 135–145")

    creatinine_phosphokinase = st.number_input("CPK (mcg/L)", 20, 8000, 250, 10,
        help="Enzyme créatinine phosphokinase · Normal : 10–120")

    platelets = st.number_input("Plaquettes (kiloplaquettes/mL)", 25000, 850000, 250000, 1000,
        help="Nombre de plaquettes · Normal : 150 000–400 000")

with col2:
    st.markdown('<div class="form-group-label">📋 Antécédents Médicaux</div>', unsafe_allow_html=True)

    anaemia = st.radio("Anémie",
        options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui",
        horizontal=True, help="Réduction des globules rouges ou hémoglobine")

    diabetes = st.radio("Diabète",
        options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui",
        horizontal=True)

    high_blood_pressure = st.radio("Hypertension artérielle",
        options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui",
        horizontal=True)

    smoking = st.radio("Tabagisme",
        options=[0, 1], format_func=lambda x: "Non" if x == 0 else "Oui",
        horizontal=True)

    sex = st.radio("Sexe",
        options=[0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme",
        horizontal=True)

    time = st.slider("Période de suivi (jours)", 4, 285, 100,
        help="Durée du suivi médical du patient")


# ============================================================
# BOUTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analyser le Risque Cardiaque", type="primary", use_container_width=True)


# ============================================================
# PRÉDICTION
# ============================================================
if predict_btn:

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

    if scaler is not None:
        patient_scaled = pd.DataFrame(
            scaler.transform(patient_df.values),
            columns=patient_df.columns
        )
    else:
        patient_scaled = patient_df

    prediction  = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    risk_pct    = probability[1] * 100
    safe_pct    = probability[0] * 100

    st.markdown("---")
    st.markdown('<div class="section-title">Résultat de l\'Analyse</div>', unsafe_allow_html=True)

    col_res, col_chart = st.columns([1, 1], gap="large")

    with col_res:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-danger">
                <div class="result-label">⚠️ Risque élevé de décès</div>
                <div class="result-number" style="color:#F56565">{risk_pct:.1f}%</div>
                <div class="result-desc">
                    Ce patient présente un <strong>risque élevé</strong> d'insuffisance cardiaque fatale.
                    Une prise en charge médicale urgente est recommandée.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-label">✅ Risque faible</div>
                <div class="result-number" style="color:#2DD4A0">{risk_pct:.1f}%</div>
                <div class="result-desc">
                    Ce patient présente un <strong>risque faible</strong> d'insuffisance cardiaque fatale.
                    Un suivi régulier est recommandé.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Probabilité Survie", f"{safe_pct:.1f}%")
        with m2:
            st.metric("Probabilité Décès", f"{risk_pct:.1f}%")

    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#0D1F3C')
        ax.set_facecolor('#112347')

        categories = ['Survie', 'Décès']
        values_bar = [safe_pct, risk_pct]
        bar_colors = ['#2DD4A0', '#F56565']

        bars = ax.barh(categories, values_bar, color=bar_colors,
                       edgecolor='none', height=0.45)

        for bar, val in zip(bars, values_bar):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center',
                    fontweight='bold', fontsize=13, color='#E8EEF8')

        ax.set_xlim(0, 115)
        ax.set_xlabel('Probabilité (%)', fontsize=10, color='#6A84A8')
        ax.tick_params(colors='#6A84A8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1E3A6E')
        ax.spines['left'].set_color('#1E3A6E')
        ax.set_title('Distribution des probabilités', fontsize=11,
                     color='#E8EEF8', fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


    # ============================================================
    # SHAP
    # ============================================================
    st.markdown("---")
    st.markdown('<div class="section-title">Explication SHAP — Pourquoi cette prédiction ?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="shap-info">
        🔴 <strong>Barre vers la droite</strong> → cette feature <strong>augmente</strong> le risque de décès &nbsp;·&nbsp;
        🔵 <strong>Barre vers la gauche</strong> → cette feature <strong>diminue</strong> le risque &nbsp;·&nbsp;
        Plus la barre est longue, plus l'impact est fort.
    </div>
    """, unsafe_allow_html=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_scaled)

        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0])
        else:
            sv = np.array(shap_values[0])

        feature_names = np.array(list(patient_df.columns))
        sv = np.array(sv).flatten()[:len(feature_names)]

        indices  = np.argsort(np.abs(sv))[::-1]
        features = list(feature_names[indices])
        values   = list(sv[indices])
        colors_s = ['#F56565' if v > 0 else '#00C2C2' for v in values]

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        fig_shap.patch.set_facecolor('#0D1F3C')
        ax_shap.set_facecolor('#112347')

        ax_shap.barh(features[::-1], values[::-1],
                     color=colors_s[::-1], edgecolor='none', height=0.6)
        ax_shap.axvline(x=0, color='#6A84A8', linewidth=1, linestyle='--', alpha=0.7)

        ax_shap.set_xlabel('Valeur SHAP (impact sur la prédiction)',
                           fontsize=11, color='#6A84A8')
        ax_shap.set_title('Impact de chaque Feature sur la Prédiction',
                          fontsize=13, fontweight='bold', color='#E8EEF8', pad=15)
        ax_shap.tick_params(colors='#B8C8E0')
        ax_shap.spines['top'].set_visible(False)
        ax_shap.spines['right'].set_visible(False)
        ax_shap.spines['bottom'].set_color('#1E3A6E')
        ax_shap.spines['left'].set_color('#1E3A6E')

        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

        st.markdown("**🏆 Top 3 features les plus influentes**")
        cols_top = st.columns(3)
        for i, col in enumerate(cols_top):
            with col:
                direction = "↑ Augmente le risque" if values[i] > 0 else "↓ Diminue le risque"
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
st.markdown("""
<div class="ecc-footer">
    <img src="https://centrale-casablanca.ma/wp-content/uploads/2023/01/Logo.png"
         style="height:30px; opacity:0.45; margin-bottom:10px; display:block; margin-left:auto; margin-right:auto;" />
    <div>
        <span class="highlight">École Centrale Casablanca</span> &nbsp;·&nbsp;
        <span class="teal">Coding Week · Mars 2026</span> &nbsp;·&nbsp;
        Heart Failure Prediction &nbsp;·&nbsp;
        <span class="yellow">Random Forest (AUC = 0.905)</span>
    </div>
    <div class="footer-team">
        <span class="footer-member">Hatim EL GAOUTI</span>
        <span class="footer-member">Adam SABILI</span>
        <span class="footer-member">Youssef ELHALLAM</span>
        <span class="footer-member">Mohamed EL YAAGOUBI</span>
        <span class="footer-member">Ilyas LESSIQ</span>
    </div>
</div>
""", unsafe_allow_html=True)