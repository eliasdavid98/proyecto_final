"""Búsqueda de hiperparámetros para los tres mejores modelos candidatos."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.svm import SVC

from src.experimentos import DEFAULT_TRACKING_URI, SCORING
from src.pipeline import make_model_pipeline
from src.preparacion import (
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    ChurnDataSplit,
    make_stratified_cv,
)

TUNING_EXPERIMENT = "pset3-churn-fine-tuning"
CONFIGURATIONS_PER_MODEL = 30


def tuning_specs(
) -> list[tuple[str, BaseEstimator, dict[str, Sequence[object]]]]:
    """Define 30 configuraciones para cada uno de los tres primeros candidatos."""
    return [
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE),
            {
                "model__n_estimators": [80, 120, 160],
                "model__learning_rate": [0.03, 0.07],
                "model__max_depth": [2, 3, 4, 5, 6],
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                max_features="sqrt",
                n_jobs=1,
                random_state=DEFAULT_RANDOM_STATE,
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 15],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
            },
        ),
        (
            "svc_rbf",
            SVC(cache_size=512, kernel="rbf"),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]


def _log_trial(
    model_name: str,
    config_number: int,
    params: dict[str, object],
    cv_results: dict[str, object],
    row_index: int,
) -> None:
    with mlflow.start_run(
        run_name=f"{model_name}_config_{config_number:02d}",
        nested=True,
    ):
        mlflow.log_params(
            {
                "model_family": model_name,
                "config_number": config_number,
                "validation": "stratified_k_fold",
                "cv_splits": DEFAULT_CV_SPLITS,
                "random_state": DEFAULT_RANDOM_STATE,
                "holdout_used": "false",
                **{name: str(value) for name, value in params.items()},
            }
        )
        for metric_name in SCORING:
            mean = float(cv_results[f"mean_test_{metric_name}"][row_index])
            std = float(cv_results[f"std_test_{metric_name}"][row_index])
            mlflow.log_metric(f"{metric_name}_mean", mean)
            mlflow.log_metric(f"{metric_name}_std", std)


def run_hyperparameter_tuning(
    split: ChurnDataSplit,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Evalúa 30 configuraciones por modelo con CV y las registra en MLflow."""
    uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    ).strip()
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI no puede estar vacío.")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(TUNING_EXPERIMENT)
    summaries: list[dict[str, str | float]] = []

    for model_name, estimator, param_grid in tuning_specs():
        configurations = list(ParameterGrid(param_grid))
        if len(configurations) != CONFIGURATIONS_PER_MODEL:
            raise ValueError(
                f"{model_name} define {len(configurations)} configuraciones; "
                f"se requieren {CONFIGURATIONS_PER_MODEL}."
            )

        with mlflow.start_run(run_name=f"{model_name}_grid_search"):
            mlflow.log_params(
                {
                    "model_family": model_name,
                    "configurations": len(configurations),
                    "cv_splits": DEFAULT_CV_SPLITS,
                    "random_state": DEFAULT_RANDOM_STATE,
                    "selection_metric": "roc_auc",
                    "development_rows": len(split.X_development),
                    "holdout_used": "false",
                }
            )
            search = GridSearchCV(
                estimator=make_model_pipeline(estimator),
                param_grid=param_grid,
                scoring=SCORING,
                refit="roc_auc",
                cv=make_stratified_cv(),
                n_jobs=1,
                pre_dispatch=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(split.X_development, split.y_development)
            cv_results = search.cv_results_

            for index, params in enumerate(cv_results["params"]):
                _log_trial(model_name, index + 1, params, cv_results, index)

            best_index = int(search.best_index_)
            summary: dict[str, str | float] = {
                "model": model_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "configurations": len(configurations),
            }
            for metric_name in SCORING:
                mean = float(cv_results[f"mean_test_{metric_name}"][best_index])
                std = float(cv_results[f"std_test_{metric_name}"][best_index])
                summary[f"{metric_name}_mean"] = mean
                summary[f"{metric_name}_std"] = std
                mlflow.log_metric(f"best_{metric_name}_mean", mean)
                mlflow.log_metric(f"best_{metric_name}_std", std)
            mlflow.log_param("best_params", summary["best_params"])
            summaries.append(summary)

    return (
        pd.DataFrame(summaries)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
