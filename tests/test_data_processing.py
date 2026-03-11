"""
test_data_processing.py
=======================
Tests automatisés pour valider toutes les fonctions de data_processing.py.

Tests couverts :
    ✅ test_load_data()          → le CSV est chargé correctement
    ✅ test_no_missing_values()  → pas de valeurs manquantes
    ✅ test_optimize_memory()    → la mémoire est bien réduite
    ✅ test_prepare_features()   → X et y sont bien séparés
    ✅ test_split_data()         → le split 80/20 est respecté
    ✅ test_normalize_data()     → moyenne ≈ 0, std ≈ 1

Usage :
    pytest tests/test_data_processing.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# ── Ajouter src/ au path pour importer data_processing ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import (
    load_data,
    optimize_memory,
    prepare_features,
    split_data,
    normalize_data,
)

# ============================================================
# FIXTURE : Dataset partagé entre tous les tests
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data',
    'heart_failure_clinical_records_dataset.csv'
)

@pytest.fixture(scope="module")
def df():
    """Charge le dataset une seule fois pour tous les tests."""
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def df_optimized(df):
    """Retourne le DataFrame après optimisation mémoire."""
    return optimize_memory(df)


# ============================================================
# TEST 1 : Chargement des données
# ============================================================

def test_load_data_shape(df):
    """Le dataset doit avoir 299 patients et 13 colonnes."""
    assert df.shape[0] == 299, f"Attendu 299 patients, obtenu {df.shape[0]}"
    assert df.shape[1] == 13,  f"Attendu 13 colonnes, obtenu {df.shape[1]}"


def test_load_data_has_target(df):
    """La colonne cible DEATH_EVENT doit exister."""
    assert 'DEATH_EVENT' in df.columns, "Colonne DEATH_EVENT manquante !"


def test_load_data_returns_dataframe(df):
    """load_data doit retourner un pd.DataFrame."""
    assert isinstance(df, pd.DataFrame)


# ============================================================
# TEST 2 : Valeurs manquantes
# ============================================================

def test_no_missing_values(df):
    """
    Le dataset ne doit contenir aucune valeur manquante (NaN).
    Ce dataset est connu pour être complet — ce test vérifie l'intégrité.
    """
    missing = df.isnull().sum().sum()
    assert missing == 0, (
        f"❌ {missing} valeur(s) manquante(s) détectée(s) !\n"
        f"{df.isnull().sum()[df.isnull().sum() > 0]}"
    )


# ============================================================
# TEST 3 : Optimisation mémoire
# ============================================================

def test_optimize_memory_reduces_size(df, df_optimized):
    """La mémoire après optimisation doit être inférieure à avant."""
    before = df.memory_usage(deep=True).sum()
    after  = df_optimized.memory_usage(deep=True).sum()
    assert after < before, (
        f"❌ Mémoire non réduite : avant={before} octets, après={after} octets"
    )


def test_optimize_memory_no_float64(df_optimized):
    """Il ne doit plus y avoir de colonnes float64 après optimisation."""
    float64_cols = df_optimized.select_dtypes(include=['float64']).columns.tolist()
    assert len(float64_cols) == 0, (
        f"❌ Colonnes float64 encore présentes : {float64_cols}"
    )


def test_optimize_memory_no_int64(df_optimized):
    """Il ne doit plus y avoir de colonnes int64 après optimisation."""
    int64_cols = df_optimized.select_dtypes(include=['int64']).columns.tolist()
    assert len(int64_cols) == 0, (
        f"❌ Colonnes int64 encore présentes : {int64_cols}"
    )


def test_optimize_memory_preserves_values(df, df_optimized):
    """Les valeurs ne doivent pas changer après la conversion des types."""
    for col in df.columns:
        assert df[col].sum() == pytest.approx(df_optimized[col].sum(), rel=1e-3), (
            f"❌ Valeurs modifiées dans la colonne '{col}'"
        )


# ============================================================
# TEST 4 : Préparation des features
# ============================================================

def test_prepare_features_splits_correctly(df_optimized):
    """X ne doit pas contenir DEATH_EVENT, y doit l'être."""
    X, y = prepare_features(df_optimized)
    assert 'DEATH_EVENT' not in X.columns, "❌ DEATH_EVENT est encore dans X !"
    assert y.name == 'DEATH_EVENT', "❌ y ne correspond pas à DEATH_EVENT"


def test_prepare_features_correct_shape(df_optimized):
    """X doit avoir 12 colonnes (13 - 1 target)."""
    X, y = prepare_features(df_optimized)
    assert X.shape[1] == 12, f"❌ X devrait avoir 12 colonnes, obtenu {X.shape[1]}"
    assert len(y) == 299,    f"❌ y devrait avoir 299 entrées, obtenu {len(y)}"


def test_prepare_features_binary_target(df_optimized):
    """DEATH_EVENT doit être binaire (seulement 0 et 1)."""
    _, y = prepare_features(df_optimized)
    unique_vals = set(y.unique())
    assert unique_vals == {0, 1}, (
        f"❌ DEATH_EVENT contient des valeurs inattendues : {unique_vals}"
    )


# ============================================================
# TEST 5 : Division Train / Test
# ============================================================

def test_split_data_proportions(df_optimized):
    """Le split doit donner ~80% train et ~20% test."""
    X, y = prepare_features(df_optimized)
    X_train, X_test, y_train, y_test = split_data(X, y)

    total = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total
    test_ratio  = len(X_test)  / total

    assert 0.78 <= train_ratio <= 0.82, f"❌ Train ratio inattendu : {train_ratio:.2f}"
    assert 0.18 <= test_ratio  <= 0.22, f"❌ Test ratio inattendu  : {test_ratio:.2f}"


def test_split_data_stratified(df_optimized):
    """
    Le ratio survivants/décédés doit être similaire dans train et test
    (grâce à stratify=y).
    """
    X, y = prepare_features(df_optimized)
    _, _, y_train, y_test = split_data(X, y)

    train_ratio = y_train.mean()
    test_ratio  = y_test.mean()

    assert abs(train_ratio - test_ratio) < 0.05, (
        f"❌ Stratification incorrecte : train={train_ratio:.2f}, test={test_ratio:.2f}"
    )


def test_split_data_no_overlap(df_optimized):
    """Les indices de train et test ne doivent pas se chevaucher."""
    X, y = prepare_features(df_optimized)
    X_train, X_test, _, _ = split_data(X, y)

    overlap = set(X_train.index) & set(X_test.index)
    assert len(overlap) == 0, f"❌ {len(overlap)} patients dans train ET test !"


# ============================================================
# TEST 6 : Normalisation
# ============================================================

def test_normalize_data_mean_near_zero(df_optimized):
    """Après normalisation, la moyenne de chaque feature doit être ≈ 0."""
    X, y = prepare_features(df_optimized)
    X_train, X_test, _, _ = split_data(X, y)
    X_train_scaled, _, _ = normalize_data(X_train, X_test)

    for col in X_train_scaled.columns:
        mean = X_train_scaled[col].mean()
        assert abs(mean) < 0.01, (
            f"❌ Moyenne non nulle pour '{col}' : {mean:.4f}"
        )


def test_normalize_data_std_near_one(df_optimized):
    """Après normalisation, l'écart-type de chaque feature doit être ≈ 1."""
    X, y = prepare_features(df_optimized)
    X_train, X_test, _, _ = split_data(X, y)
    X_train_scaled, _, _ = normalize_data(X_train, X_test)

    for col in X_train_scaled.columns:
        std = X_train_scaled[col].std()
        assert abs(std - 1.0) < 0.05, (
            f"❌ Std non unitaire pour '{col}' : {std:.4f}"
        )


def test_normalize_preserves_shape(df_optimized):
    """La normalisation ne doit pas changer le nombre de lignes/colonnes."""
    X, y = prepare_features(df_optimized)
    X_train, X_test, _, _ = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = normalize_data(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape  == X_test.shape