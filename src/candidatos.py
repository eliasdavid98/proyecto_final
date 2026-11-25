"""Modelos candidatos para comparar en el split de desarrollo."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.experimentos import evaluate_estimators
from src.preparacion import DEFAULT_RANDOM_STATE, ChurnDataSplit

CANDIDATE_EXPERIMENT = "pset3-churn-candidates"


def candidate_estimators() -> list[tuple[str, BaseEstimator]]:
    """Devuelve cinco familias distintas con configuraciones iniciales."""
    return [
        (
            "decision_tree",
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=20,
                random_state=DEFAULT_RANDOM_STATE,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                max_features="sqrt",
                min_samples_leaf=5,
                n_estimators=150,
                n_jobs=1,
                random_state=DEFAULT_RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                n_estimators=100,
                random_state=DEFAULT_RANDOM_STATE,
            ),
        ),
        (
            "svc_rbf",
            SVC(
                C=1.0,
                class_weight="balanced",
                kernel="rbf",
            ),
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_jobs=1,
                n_neighbors=15,
                weights="distance",
            ),
        ),
    ]


def run_candidate_models(
    split: ChurnDataSplit,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Evalúa los cinco candidatos y registra sus resultados en MLflow."""
    return evaluate_estimators(
        split=split,
        estimators=candidate_estimators(),
        experiment_name=CANDIDATE_EXPERIMENT,
        tracking_uri=tracking_uri,
    )
