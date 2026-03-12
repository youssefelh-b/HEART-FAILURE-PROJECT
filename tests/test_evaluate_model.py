"""
test_evaluate_model.py
======================
Tests automatises pour valider les fonctions de evaluate_model.py.

Tests couverts :
    test_get_test_data()                     -> recupere bien X_test et y_test du pipeline
    test_evaluate_one_model_with_proba()    -> calcule correctement les metriques classiques
    test_evaluate_one_model_without_proba() -> utilise bien le fallback predict()
    test_evaluate_one_model_missing_file()  -> un fichier absent renvoie None
    test_print_comparison_table()           -> retourne un DataFrame propre
    test_plot_metrics_comparison()          -> sauvegarde les 3 graphiques attendus
    test_choose_best_model()                -> choisit et sauvegarde le meilleur modele

Usage :
    pytest tests/test_evaluate_model.py -v
"""

import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

# Ajouter src/ au path pour importer evaluate_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import evaluate_model
from evaluate_model import choose_best_model, evaluate_one_model, get_test_data, plot_metrics_comparison, print_comparison_table


class FakeProbaModel:
    """Modele de test avec predict et predict_proba."""

    def predict(self, X_test):
        return np.array([0, 1, 1, 0])

    def predict_proba(self, X_test):
        return np.array([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.15, 0.85],
            [0.70, 0.30],
        ])


class FakeScoreModel:
    """Modele de test sans predict_proba, type booster natif."""

    def predict(self, X_test):
        return np.array([0.10, 0.80, 0.65, 0.30])


# ============================================================
# FIXTURES : Donnees partagees entre les tests
# ============================================================

@pytest.fixture(scope="module")
def sample_test_data():
    """Jeu de test minimal, stable et entierement deterministe."""
    X_test = pd.DataFrame(
        {
            f"feature_{index}": [index, index + 1, index + 2, index + 3]
            for index in range(12)
        }
    )
    y_test = pd.Series([0, 1, 1, 0], name="DEATH_EVENT")
    return X_test, y_test


@pytest.fixture(scope="module")
def comparison_results(sample_test_data):
    """Resultats factices coherents pour tester tableau, graphiques et selection."""
    _, y_test = sample_test_data
    y_true = y_test.to_numpy()
    return [
        {
            'Model': 'Logistic Regression',
            'ROC-AUC': 0.9100,
            'Accuracy': 0.8500,
            'Precision': 0.8200,
            'Recall': 0.8800,
            'F1-Score': 0.8500,
            '_y_proba': np.array([0.10, 0.80, 0.75, 0.20]),
            '_y_pred': np.array([0, 1, 1, 0]),
            '_y_test': y_true,
        },
        {
            'Model': 'Random Forest',
            'ROC-AUC': 0.8600,
            'Accuracy': 0.8000,
            'Precision': 0.7800,
            'Recall': 0.8200,
            'F1-Score': 0.8000,
            '_y_proba': np.array([0.20, 0.75, 0.70, 0.35]),
            '_y_pred': np.array([0, 1, 1, 0]),
            '_y_test': y_true,
        },
        {
            'Model': 'LightGBM',
            'ROC-AUC': 0.9300,
            'Accuracy': 0.8700,
            'Precision': 0.8400,
            'Recall': 0.9000,
            'F1-Score': 0.8700,
            '_y_proba': np.array([0.05, 0.88, 0.81, 0.18]),
            '_y_pred': np.array([0, 1, 1, 0]),
            '_y_test': y_true,
        },
        {
            'Model': 'XGBoost',
            'ROC-AUC': 0.8900,
            'Accuracy': 0.8300,
            'Precision': 0.8000,
            'Recall': 0.8600,
            'F1-Score': 0.8300,
            '_y_proba': np.array([0.12, 0.79, 0.76, 0.22]),
            '_y_pred': np.array([0, 1, 1, 0]),
            '_y_test': y_true,
        },
    ]


@pytest.fixture(scope="module")
def comparison_df(comparison_results):
    """Tableau de comparaison partage pour plusieurs tests."""
    return print_comparison_table(comparison_results)


# ============================================================
# TEST 1 : Recuperation des donnees de test
# ============================================================

def test_get_test_data_returns_pipeline_test_split(sample_test_data, monkeypatch):
    """get_test_data doit renvoyer exactement X_test et y_test issus du pipeline."""
    X_test, y_test = sample_test_data
    X_train = pd.DataFrame({'feature_0': [0, 1]})
    y_train = pd.Series([0, 1], name='DEATH_EVENT')
    scaler = object()
    called = {}

    def fake_run_pipeline(data_path):
        called['data_path'] = data_path
        return X_train, X_test, y_train, y_test, scaler

    monkeypatch.setattr(evaluate_model, 'run_pipeline', fake_run_pipeline)

    returned_X_test, returned_y_test = get_test_data()

    assert called['data_path'] == evaluate_model.DATA_PATH
    pd.testing.assert_frame_equal(returned_X_test, X_test)
    pd.testing.assert_series_equal(returned_y_test, y_test)


# ============================================================
# TEST 2 : Evaluation d'un modele
# ============================================================

def test_evaluate_one_model_with_predict_proba(sample_test_data, tmp_path, monkeypatch):
    """Un modele sklearn classique doit retourner toutes les metriques attendues."""
    X_test, y_test = sample_test_data
    model_path = tmp_path / 'model_with_proba.pkl'
    model_path.write_text('placeholder', encoding='utf-8')

    monkeypatch.setattr(joblib, 'load', lambda path: {'model': FakeProbaModel()})

    metrics = evaluate_one_model('Test Model', os.fspath(model_path), X_test, y_test)

    assert metrics['Model'] == 'Test Model'
    assert metrics['ROC-AUC'] == pytest.approx(1.0)
    assert metrics['Accuracy'] == pytest.approx(1.0)
    assert metrics['Precision'] == pytest.approx(1.0)
    assert metrics['Recall'] == pytest.approx(1.0)
    assert metrics['F1-Score'] == pytest.approx(1.0)
    assert np.array_equal(metrics['_y_pred'], np.array([0, 1, 1, 0]))
    assert np.array_equal(metrics['_y_proba'], np.array([0.10, 0.80, 0.85, 0.30]))


def test_evaluate_one_model_without_predict_proba_uses_predict_scores(sample_test_data, tmp_path, monkeypatch):
    """Un modele sans predict_proba doit utiliser predict() comme score et seuiller a 0.5."""
    X_test, y_test = sample_test_data
    model_path = tmp_path / 'model_without_proba.pkl'
    model_path.write_text('placeholder', encoding='utf-8')

    monkeypatch.setattr(joblib, 'load', lambda path: FakeScoreModel())

    metrics = evaluate_one_model('Booster Model', os.fspath(model_path), X_test, y_test)

    assert metrics['Model'] == 'Booster Model'
    assert metrics['ROC-AUC'] == pytest.approx(1.0)
    assert metrics['Accuracy'] == pytest.approx(1.0)
    assert metrics['Recall'] == pytest.approx(1.0)
    assert np.array_equal(metrics['_y_pred'], np.array([0, 1, 1, 0]))
    assert np.array_equal(metrics['_y_proba'], np.array([0.10, 0.80, 0.65, 0.30]))


def test_evaluate_one_model_missing_file_returns_none(sample_test_data, tmp_path):
    """Un chemin de modele inexistant doit renvoyer None."""
    X_test, y_test = sample_test_data
    missing_path = tmp_path / 'missing_model.pkl'
    assert evaluate_one_model('Missing Model', os.fspath(missing_path), X_test, y_test) is None


# ============================================================
# TEST 3 : Tableau de comparaison
# ============================================================

def test_print_comparison_table_returns_dataframe(comparison_df):
    """Le tableau de comparaison doit contenir 4 lignes et 5 metriques visibles."""
    assert isinstance(comparison_df, pd.DataFrame)
    assert comparison_df.shape == (4, 5)
    assert list(comparison_df.columns) == ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    assert sorted(comparison_df.index.tolist()) == [
        'LightGBM',
        'Logistic Regression',
        'Random Forest',
        'XGBoost',
    ]


# ============================================================
# TEST 4 : Graphiques d'evaluation
# ============================================================

def test_plot_metrics_comparison_saves_expected_files(comparison_df, comparison_results, tmp_path, monkeypatch):
    """Les 3 graphiques doivent etre sauvegardes dans RESULTS_DIR."""
    monkeypatch.setattr(evaluate_model, 'RESULTS_DIR', os.fspath(tmp_path))
    monkeypatch.setattr(evaluate_model.plt, 'show', lambda: None)

    plot_metrics_comparison(comparison_df.copy(), comparison_results)

    assert (tmp_path / 'metrics_comparison.png').exists()
    assert (tmp_path / 'roc_curves.png').exists()
    assert (tmp_path / 'confusion_matrices.png').exists()


# ============================================================
# TEST 5 : Choix du meilleur modele
# ============================================================

def test_choose_best_model_selects_and_saves_best_artifact(comparison_df, comparison_results, tmp_path, monkeypatch):
    """Le meilleur modele doit etre LightGBM et etre sauvegarde en best_model.pkl."""
    loaded_paths = []
    dumped = {}

    def fake_load(path):
        loaded_paths.append(path)
        return {'model': 'serialized-lightgbm'}

    def fake_dump(data, path):
        dumped['data'] = data
        dumped['path'] = path

    monkeypatch.setattr(evaluate_model, 'MODELS_DIR', os.fspath(tmp_path))
    monkeypatch.setattr(joblib, 'load', fake_load)
    monkeypatch.setattr(joblib, 'dump', fake_dump)

    best_model = choose_best_model(comparison_df.copy(), comparison_results)

    assert best_model == 'LightGBM'
    assert loaded_paths == [os.path.join(os.fspath(tmp_path), 'lightgbm.pkl')]
    assert dumped['data'] == {'model': 'serialized-lightgbm'}
    assert dumped['path'] == os.path.join(os.fspath(tmp_path), 'best_model.pkl')
