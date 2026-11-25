"""Evaluación final en holdout y registro del modelo ganador en MLflow."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.candidatos import candidate_estimators
from src.experimentos import DEFAULT_TRACKING_URI
from src.features import ChurnFeatureEngineer
from src.pipeline import make_model_pipeline
from src.preparacion import ChurnDataSplit

TUNING_EXPERIMENT = "pset3-churn-fine-tuning"
FINAL_EXPERIMENT = "pset3-churn-final-holdout"
REGISTERED_MODEL_NAME = "pset3-churn-classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
SKOPS_TRUSTED_TYPES = (
    "numpy.dtype",
    "numpy.number",
    "sklearn.compose._column_transformer.make_column_selector",
    "sklearn.tree._tree.Tree",
    "src.features.ChurnFeatureEngineer",
)


@dataclass(frozen=True)
class FinalistEvaluation:
    model: str
    best_cv_roc_auc: float
    best_params: dict[str, object]
    holdout_metrics: dict[str, float]
    estimator: Pipeline


def _top_tuned_finalists() -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(TUNING_EXPERIMENT)
    if experiment is None:
        raise ValueError(
            "No existe la búsqueda de hiperparámetros; ejecuta primero el paso 7."
        )

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    parents = runs.loc[
        runs["params.configurations"].notna()
        & runs["params.best_params"].notna()
        & runs["status"].eq("FINISHED")
    ]
    finalists = (
        parents.sort_values("metrics.best_roc_auc_mean", ascending=False)
        .drop_duplicates("params.model_family")
        .head(2)
    )
    if len(finalists) != 2:
        raise ValueError(
            f"Se requieren dos búsquedas de modelo completas; se encontraron {len(finalists)}."
        )
    return finalists


def _ensure_holdout_unused() -> None:
    experiment = mlflow.get_experiment_by_name(FINAL_EXPERIMENT)
    if experiment is None:
        return
    if experiment.tags.get("holdout_evaluation_started") == "true":
        raise RuntimeError(
            "La evaluación final del holdout ya se inició. Revisa MLflow; "
            "no lo vuelvas a evaluar."
        )
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if "params.holdout_used" in runs and runs["params.holdout_used"].eq("true").any():
        raise RuntimeError(
            "El holdout ya fue usado en la evaluación final. Revisa MLflow; "
            "no lo vuelvas a evaluar."
        )


def _holdout_metrics(
    estimator: Pipeline,
    split: ChurnDataSplit,
) -> dict[str, float]:
    predictions = estimator.predict(split.X_holdout)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(split.X_holdout)
        positive_index = list(estimator.classes_).index(1)
        scores = probabilities[:, positive_index]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(split.X_holdout)
    else:
        raise TypeError("El estimador final debe exponer predict_proba o decision_function.")

    return {
        "roc_auc": float(roc_auc_score(split.y_holdout, scores)),
        "average_precision": float(
            average_precision_score(split.y_holdout, scores)
        ),
        "f1": float(f1_score(split.y_holdout, predictions, zero_division=0)),
        "precision": float(
            precision_score(split.y_holdout, predictions, zero_division=0)
        ),
        "recall": float(recall_score(split.y_holdout, predictions, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(split.y_holdout, predictions)
        ),
    }


def _model_input_example(frame: pd.DataFrame) -> pd.DataFrame:
    example = frame.head(5).copy()
    for column in ChurnFeatureEngineer.numeric_columns:
        if column in example:
            example[column] = pd.to_numeric(
                example[column], errors="coerce"
            ).astype("float64")
    return example


def evaluate_finalists_and_register_champion(
    split: ChurnDataSplit,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Evalúa una vez los dos finalistas y registra el de mayor ROC-AUC holdout."""
    uri = (
        tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    ).strip()
    if not uri:
        raise ValueError("MLFLOW_TRACKING_URI no puede estar vacío.")

    mlflow.set_tracking_uri(uri)
    _ensure_holdout_unused()
    finalists = _top_tuned_finalists()
    base_estimators = dict(candidate_estimators())
    prepared: list[tuple[str, float, dict[str, object], Pipeline]] = []

    for _, tuning_run in finalists.iterrows():
        model_name = tuning_run["params.model_family"]
        if model_name not in base_estimators:
            raise ValueError(f"No existe una configuración base para {model_name}.")
        best_params = json.loads(tuning_run["params.best_params"])
        estimator = make_model_pipeline(base_estimators[model_name])
        estimator.set_params(**best_params)
        estimator.fit(split.X_development, split.y_development)
        prepared.append(
            (
                model_name,
                float(tuning_run["metrics.best_roc_auc_mean"]),
                best_params,
                estimator,
            )
        )

    active_experiment = mlflow.set_experiment(FINAL_EXPERIMENT)
    client = MlflowClient()
    client.set_experiment_tag(
        active_experiment.experiment_id,
        "holdout_evaluation_started",
        "true",
    )
    evaluated: list[FinalistEvaluation] = []
    for model_name, best_cv_roc_auc, best_params, estimator in prepared:
        metrics = _holdout_metrics(estimator, split)
        evaluated.append(
            FinalistEvaluation(
                model=model_name,
                best_cv_roc_auc=best_cv_roc_auc,
                best_params=best_params,
                holdout_metrics=metrics,
                estimator=estimator,
            )
        )

    winner = max(
        evaluated,
        key=lambda result: (
            result.holdout_metrics["roc_auc"],
            result.holdout_metrics["average_precision"],
        ),
    )
    source_directory = Path(__file__).resolve().parent
    finalist_run_ids: dict[str, str] = {}

    for result in evaluated:
        is_winner = result.model == winner.model
        with mlflow.start_run(run_name=f"{result.model}_holdout_final") as run:
            mlflow.log_params(
                {
                    "model_family": result.model,
                    "selection_metric": "roc_auc",
                    "best_cv_roc_auc": result.best_cv_roc_auc,
                    "best_params": json.dumps(result.best_params, sort_keys=True),
                    "holdout_used": "true",
                    "development_rows": len(split.X_development),
                    "holdout_rows": len(split.X_holdout),
                    "selected_as_champion": str(is_winner).lower(),
                }
            )
            mlflow.log_metrics(
                {
                    f"holdout_{name}": value
                    for name, value in result.holdout_metrics.items()
                }
            )
            finalist_run_ids[result.model] = run.info.run_id

    input_example = _model_input_example(split.X_development)
    prediction_example = winner.estimator.predict(input_example)
    signature = infer_signature(input_example, prediction_example)
    with mlflow.start_run(run_name=f"{winner.model}_champion_registration") as run:
        mlflow.log_params(
            {
                "model_family": winner.model,
                "selection_metric": "roc_auc",
                "selected_as_champion": "true",
                "holdout_used": "true",
                "best_params": json.dumps(winner.best_params, sort_keys=True),
                "selection_run_id": finalist_run_ids[winner.model],
            }
        )
        mlflow.log_metrics(
            {
                f"holdout_{name}": value
                for name, value in winner.holdout_metrics.items()
            }
        )
        mlflow.sklearn.log_model(
            sk_model=winner.estimator,
            artifact_path="model",
            code_paths=[str(source_directory)],
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ],
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
            await_registration_for=300,
        )
        winner_run_id = run.info.run_id

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )
    registered_version = next(
        (
            version
            for version in versions
            if version.run_id == winner_run_id
        ),
        None,
    )
    if registered_version is None:
        raise RuntimeError("MLflow no devolvió la versión recién registrada.")
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHAMPION_ALIAS,
        registered_version.version,
    )

    rows = []
    for result in evaluated:
        row = {
            "model": result.model,
            "best_cv_roc_auc": result.best_cv_roc_auc,
            "best_params": json.dumps(result.best_params, sort_keys=True),
            **{
                f"holdout_{name}": value
                for name, value in result.holdout_metrics.items()
            },
            "champion": result.model == winner.model,
        }
        if row["champion"]:
            row["registry_model"] = REGISTERED_MODEL_NAME
            row["registry_version"] = registered_version.version
            row["registry_alias"] = CHAMPION_ALIAS
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "holdout_roc_auc", ascending=False
    ).reset_index(drop=True)
