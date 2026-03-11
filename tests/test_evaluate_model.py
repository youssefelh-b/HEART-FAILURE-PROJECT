"""Tests for LightGBM evaluation metrics only (no model training)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate_model import evaluate_metrics


def test_evaluate_metrics_returns_required_keys_and_range():
    """Evaluation should return the required metric keys in valid range [0, 1]."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.10, 0.90, 0.20, 0.40, 0.80, 0.30, 0.95, 0.55])

    metrics = evaluate_metrics(y_true, y_pred, y_pred_proba)

    required_metrics = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert required_metrics.issubset(metrics.keys())

    for key in required_metrics:
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_metrics_supports_two_dimensional_probabilities():
    """Evaluation should support predict_proba-like arrays of shape (n, 2)."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_pred_proba_2d = np.array(
        [
            [0.85, 0.15],
            [0.10, 0.90],
            [0.75, 0.25],
            [0.20, 0.80],
            [0.55, 0.45],
            [0.60, 0.40],
        ]
    )

    metrics = evaluate_metrics(y_true, y_pred, y_pred_proba_2d)

    assert metrics["roc_auc"] >= 0.0
    assert metrics["roc_auc"] <= 1.0


def test_evaluate_metrics_with_zero_division_case():
    """Evaluation should handle precision/recall zero-division cases safely."""
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.10, 0.20, 0.05, 0.30])

    metrics = evaluate_metrics(y_true, y_pred, y_pred_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
