"""Evaluation helpers for binary classification."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_metrics(y_true, y_pred, y_pred_proba):
	"""Compute core metrics requested for model evaluation."""
	if y_pred_proba.ndim == 2:
		positive_proba = y_pred_proba[:, 1]
	else:
		positive_proba = y_pred_proba

	return {
		"roc_auc": roc_auc_score(y_true, positive_proba),
		"accuracy": accuracy_score(y_true, y_pred),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
		"f1": f1_score(y_true, y_pred, zero_division=0),
	}


def print_metrics(metrics: dict) -> None:
	"""Affiche les metriques dans le meme style visuel que data_processing.py."""
	print("\n" + "=" * 50)
	print("📊 EVALUATION DU MODELE LIGHTGBM")
	print("=" * 50)
	print(f"✅ ROC-AUC    : {metrics['roc_auc']:.4f}")
	print(f"✅ Accuracy   : {metrics['accuracy']:.4f}")
	print(f"✅ Precision  : {metrics['precision']:.4f}")
	print(f"✅ Recall     : {metrics['recall']:.4f}")
	print(f"✅ F1-score   : {metrics['f1']:.4f}")
	print("=" * 50)
