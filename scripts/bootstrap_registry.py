"""Importa las versiones champion/challenger existentes al MLflow de Compose."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seleccion import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    SKOPS_TRUSTED_TYPES,
)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
FIXTURE_DIR = ROOT / "tests" / "fixtures"
MODEL_SPECS = (
    ("champion", CHAMPION_ALIAS),
    ("challenger", CHALLENGER_ALIAS),
)


def _read_model_sample() -> tuple[pd.DataFrame, dict[str, object]]:
    metadata = json.loads((FIXTURE_DIR / "champion_sample.json").read_text())
    return pd.DataFrame(metadata["features"]), metadata


def import_registered_versions() -> dict[str, str]:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = mlflow.set_experiment("pset4-model-import")
    frame, metadata = _read_model_sample()
    versions: dict[str, str] = {}

    for model_dir, alias in MODEL_SPECS:
        try:
            existing = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME, alias
            )
        except MlflowException as error:
            missing_alias = (
                error.error_code == "INVALID_PARAMETER_VALUE"
                and f"registered model alias {alias.lower()} not found"
                in str(error).lower()
            )
            if error.error_code != "RESOURCE_DOES_NOT_EXIST" and not missing_alias:
                raise
            existing = None

        if existing is not None:
            versions[alias] = str(existing.version)
            continue

        fixture_path = FIXTURE_DIR / f"{model_dir}_model"
        model = mlflow.sklearn.load_model(str(fixture_path))
        example = frame.head(5).copy()
        signature = infer_signature(example, model.predict(example))
        source_version = metadata[
            "version" if alias == CHAMPION_ALIAS else "challenger_version"
        ]
        source_run_id = metadata[
            "run_id" if alias == CHAMPION_ALIAS else "challenger_run_id"
        ]
        experiment_id = experiment.experiment_id
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"import_{alias}_from_pset3_v{source_version}",
        ) as run:
            mlflow.log_params(
                {
                    "model_role": alias,
                    "source_registry": "PSet3 local registry export",
                    "source_version": source_version,
                    "source_run_id": source_run_id,
                }
            )
            cv_auc = (
                metadata["reference_metrics_from_registry_run"]["cv_roc_auc"]
                if alias == CHAMPION_ALIAS
                else 0.923759091400825
            )
            mlflow.log_metric("source_cv_roc_auc", float(cv_auc))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                code_paths=[str(ROOT / "src")],
                signature=signature,
                input_example=example,
                registered_model_name=REGISTERED_MODEL_NAME,
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"pandas=={pd.__version__}",
                ],
                skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
                await_registration_for=300,
            )
            run_id = run.info.run_id

        registered = next(
            (
                version
                for version in client.search_model_versions(
                    f"name='{REGISTERED_MODEL_NAME}'"
                )
                if version.run_id == run_id
            ),
            None,
        )
        if registered is None:
            raise RuntimeError(f"No se encontró versión registrada para {alias}.")
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, alias, registered.version
        )
        versions[alias] = str(registered.version)

    return versions


if __name__ == "__main__":
    print(json.dumps(import_registered_versions(), indent=2))
