"""Registra el mejor Random Forest ajustado como alias challenger."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidatos import candidate_estimators
from src.experimentos import DEFAULT_TRACKING_URI
from src.pipeline import make_model_pipeline
from src.preparacion import split_churn_data
from src.seleccion import (
    CHALLENGER_ALIAS,
    REGISTERED_MODEL_NAME,
    SKOPS_TRUSTED_TYPES,
    _model_input_example,
)
from src.datos import load_obt


TUNING_EXPERIMENT = "pset3-churn-fine-tuning"


def _best_random_forest_run() -> pd.Series:
    experiment = mlflow.get_experiment_by_name(TUNING_EXPERIMENT)
    if experiment is None:
        raise ValueError("No existe el experimento de fine-tuning.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    parents = runs.loc[
        runs["params.model_family"].eq("random_forest")
        & runs["params.configurations"].notna()
        & runs["params.best_params"].notna()
        & runs["status"].eq("FINISHED")
    ]
    if len(parents) != 1:
        raise ValueError(
            "Se esperaba una corrida de búsqueda completa para Random Forest; "
            f"se encontraron {len(parents)}."
        )
    return parents.iloc[0]


def register_challenger(env_file: Path, tracking_uri: str) -> str:
    if not env_file.is_absolute():
        env_file = ROOT / env_file
    load_dotenv(env_file)
    mlflow.set_tracking_uri(tracking_uri)

    tuning_run = _best_random_forest_run()
    best_params = json.loads(tuning_run["params.best_params"])
    estimators = dict(candidate_estimators())
    if "random_forest" not in estimators:
        raise ValueError("No existe el estimador base de Random Forest.")

    obt = load_obt()
    obt.columns = obt.columns.str.lower()
    split = split_churn_data(obt)
    estimator = make_model_pipeline(estimators["random_forest"])
    estimator.set_params(**best_params)
    estimator.fit(split.X_development, split.y_development)

    input_example = _model_input_example(split.X_development)
    signature = infer_signature(input_example, estimator.predict(input_example))
    client = MlflowClient()
    experiment = mlflow.set_experiment("pset3-churn-final-holdout")

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="random_forest_challenger_registration",
    ) as run:
        mlflow.log_params(
            {
                "model_family": "random_forest",
                "model_role": CHALLENGER_ALIAS,
                "selected_as_challenger": "true",
                "holdout_used": "false",
                "best_params": json.dumps(best_params, sort_keys=True),
                "selection_source_run_id": tuning_run["run_id"],
                "development_rows": len(split.X_development),
            }
        )
        mlflow.log_metrics(
            {
                "best_cv_roc_auc": float(
                    tuning_run["metrics.best_roc_auc_mean"]
                ),
                "best_cv_average_precision": float(
                    tuning_run["metrics.best_average_precision_mean"]
                ),
            }
        )
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            code_paths=[str(ROOT / "src")],
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
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = next((item for item in versions if item.run_id == run_id), None)
    if version is None:
        raise RuntimeError("MLflow no devolvió la versión challenger recién registrada.")

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        CHALLENGER_ALIAS,
        version.version,
    )
    return version.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Archivo con credenciales Snowflake (por defecto PSet3/.env).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="URI del servidor local de MLflow.",
    )
    args = parser.parse_args()
    version = register_challenger(args.env_file, args.tracking_uri)
    print(
        f"{REGISTERED_MODEL_NAME}@{CHALLENGER_ALIAS} "
        f"apunta a la versión {version}."
    )


if __name__ == "__main__":
    main()
