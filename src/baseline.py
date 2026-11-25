"""Entrenamiento y registro de líneas base con validación cruzada."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.experimentos import evaluate_estimators
from src.preparacion import (
    DEFAULT_RANDOM_STATE,
    ChurnDataSplit,
)

BASELINE_EXPERIMENT = "pset3-churn-baselines"


def _baseline_estimators() -> list[tuple[str, BaseEstimator]]:
    return [
        ("dummy_prior", DummyClassifier(strategy="prior")),
        (
            "logistic_regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=DEFAULT_RANDOM_STATE,
            ),
        ),
    ]


def run_baselines(
    split: ChurnDataSplit,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Evalúa líneas base solo en desarrollo y registra CV en el servidor MLflow."""
    return evaluate_estimators(
        split=split,
        estimators=_baseline_estimators(),
        experiment_name=BASELINE_EXPERIMENT,
        tracking_uri=tracking_uri,
    )
