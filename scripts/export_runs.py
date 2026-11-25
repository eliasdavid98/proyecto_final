"""Exporta todas las corridas de experimentos PSet 3 desde MLflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from pathlib import PurePosixPath

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experimentos import DEFAULT_TRACKING_URI


OUTPUT_PATH = ROOT / "docs" / "corridas.csv"
RETRAINING_OUTPUT_PATH = ROOT / "docs" / "reentrenamientos.csv"
EXPERIMENT_PREFIXES = ("pset3-churn-", "pset4-churn-")


def _portable_artifact_uri(experiment_id: object, run_id: object) -> str:
    """Return a project-relative artifact path without machine-specific roots."""
    return PurePosixPath(
        "mlartifacts",
        str(experiment_id),
        str(run_id),
        "artifacts",
    ).as_posix()


def export_runs(tracking_uri: str = DEFAULT_TRACKING_URI) -> int:
    mlflow.set_tracking_uri(tracking_uri)
    experiments = [
        experiment
        for experiment in MlflowClient().search_experiments()
        if experiment.name.startswith(EXPERIMENT_PREFIXES)
    ]
    if not experiments:
        raise RuntimeError(
            f"No se encontraron experimentos con prefijos {EXPERIMENT_PREFIXES!r}."
        )

    frames = []
    for experiment in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            output_format="pandas",
        )
        if not runs.empty:
            runs.insert(0, "experiment_name", experiment.name)
            frames.append(runs)
    if not frames:
        raise RuntimeError("Los experimentos encontrados no contienen corridas.")

    exported = pd.concat(frames, ignore_index=True, sort=False)
    if {"artifact_uri", "experiment_id", "run_id"}.issubset(exported.columns):
        exported["artifact_uri"] = [
            _portable_artifact_uri(experiment_id, run_id)
            if pd.notna(artifact_uri)
            else artifact_uri
            for artifact_uri, experiment_id, run_id in zip(
                exported["artifact_uri"],
                exported["experiment_id"],
                exported["run_id"],
                strict=True,
            )
        ]
    if "start_time" in exported:
        exported = exported.sort_values(
            ["experiment_name", "start_time"],
            kind="stable",
            na_position="last",
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(OUTPUT_PATH, index=False)
    phase_columns = [
        column
        for column in ("tags.fase", "params.fase")
        if column in exported.columns
    ]
    if phase_columns:
        retraining_mask = pd.Series(False, index=exported.index)
        for column in phase_columns:
            retraining_mask |= exported[column].eq("reentrenamiento")
        retraining = exported.loc[retraining_mask]
    else:
        retraining = exported.iloc[0:0]
    retraining.to_csv(RETRAINING_OUTPUT_PATH, index=False)
    print(f"Exportadas {len(exported)} corridas a {OUTPUT_PATH}")
    print(
        f"Exportadas {len(retraining)} corridas de reentrenamiento "
        f"a {RETRAINING_OUTPUT_PATH}"
    )
    return len(exported)


if __name__ == "__main__":
    export_runs(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
