"""Evaluación común y registro MLflow para modelos de churn."""

from __future__ import annotations

import os
from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

from src.preparacion import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    ChurnDataSplit,
    make_stratified_cv,
)
from src.pipeline import make_model_pipeline

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
}


def evaluate_estimators(
    split: ChurnDataSplit,
    estimators: Sequence[tuple[str, BaseEstimator]],
    experiment_name: str,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Evalúa estimadores con la misma CV y registra parámetros/métricas en MLflow."""
    uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    ).strip()
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI no puede estar vacío.")
    if not experiment_name.strip():
        raise ValueError("experiment_name no puede estar vacío.")
    if not estimators:
        raise ValueError("Se requiere al menos un estimador para evaluar.")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    cv = make_stratified_cv()
    summaries: list[dict[str, float | str]] = []

    for model_name, estimator in estimators:
        with mlflow.start_run(run_name=model_name):
            params = {
                "model": model_name,
                "validation": "stratified_k_fold",
                "cv_splits": DEFAULT_CV_SPLITS,
                "random_state": DEFAULT_RANDOM_STATE,
                "development_rows": len(split.X_development),
                "holdout_used": "false",
            }
            params.update(
                {
                    f"estimator_{name}": str(value)
                    for name, value in estimator.get_params(deep=False).items()
                }
            )
            mlflow.log_params(params)

            cv_results = cross_validate(
                estimator=make_model_pipeline(estimator),
                X=split.X_development,
                y=split.y_development,
                cv=cv,
                scoring=SCORING,
                n_jobs=1,
                error_score="raise",
            )
            summary: dict[str, float | str] = {"model": model_name}
            for metric_name in SCORING:
                values = cv_results[f"test_{metric_name}"]
                mean = float(np.mean(values))
                std = float(np.std(values))
                summary[f"{metric_name}_mean"] = mean
                summary[f"{metric_name}_std"] = std
                mlflow.log_metric(f"{metric_name}_mean", mean)
                mlflow.log_metric(f"{metric_name}_std", std)
            summaries.append(summary)

    return (
        pd.DataFrame(summaries)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
