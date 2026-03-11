"""Evaluation helpers for binary classification."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from data_processing import run_pipeline


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


def run_lightgbm_evaluation() -> None:
	"""Train/evaluate LightGBM and print metrics in terminal."""
	project_root = Path(__file__).resolve().parents[1]
	data_path = project_root / "data" / "heart_failure_clinical_records_dataset.csv"

	X_train, X_test, y_train, y_test, _ = run_pipeline(str(data_path))

	model = lgb.LGBMClassifier(
		n_estimators=100,
		learning_rate=0.1,
		max_depth=5,
		num_leaves=31,
		random_state=42,
		verbose=-1,
	)

	model.fit(
		X_train,
		y_train,
		eval_set=[(X_test, y_test)],
		eval_metric="auc",
		callbacks=[lgb.early_stopping(10, verbose=False)],
	)

	y_pred = model.predict(X_test)
	y_pred_proba = model.predict_proba(X_test)

	metrics = evaluate_metrics(y_test, y_pred, y_pred_proba)
	print_metrics(metrics)


if __name__ == "__main__":
	run_lightgbm_evaluation()
