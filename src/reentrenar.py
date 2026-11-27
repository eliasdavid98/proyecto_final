"""Tubería de reentrenamiento, evaluación champion/challenger y promoción."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import snowflake.connector
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import make_model_pipeline
from src.seleccion import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    SKOPS_TRUSTED_TYPES,
)

MODEL_INPUTS = (
    "geography",
    "gender",
    "age_band",
    "age_band_min_age",
    "age_band_max_age",
    "number_of_products",
    "credit_score",
    "age",
    "tenure",
    "balance",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
)
FLOAT_COLUMNS = (
    "number_of_products",
    "credit_score",
    "age",
    "tenure",
    "balance",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
)
# Hiperparámetros congelados del champion PSet #3 (sin repetir tuning)
CHAMPION_HYPERPARAMS = {
    "learning_rate": 0.07,
    "max_depth": 4,
    "n_estimators": 120,
    "random_state": 42,
}
CV_ROC_AUC_STD = 0.0026
DECISION_THRESHOLD = float(os.getenv("CHURN_DECISION_THRESHOLD", "0.5"))
CONTACT_COST_USD = float(os.getenv("CONTACT_COST_USD", "25.0"))
RETENTION_VALUE_USD = float(os.getenv("RETENTION_VALUE_USD", "300.0"))
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable requerida {name}.")
    return value


def _identifier(name: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} debe ser un identificador Snowflake válido.")
    return value.upper()


def _snowflake_connection() -> Any:
    options: dict[str, Any] = {
        "account": _required_env("SNOWFLAKE_ACCOUNT"),
        "user": _required_env("SNOWFLAKE_USERNAME"),
        "password": _required_env("SNOWFLAKE_PASSWORD"),
        "database": _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE")),
        "warehouse": _required_env("SNOWFLAKE_WAREHOUSE"),
        "login_timeout": 10,
    }
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        options["role"] = role
    return snowflake.connector.connect(**options)


def _prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(MODEL_INPUTS)].copy()
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    out["age_band_min_age"] = pd.to_numeric(out["age_band_min_age"], errors="raise").astype("int64")
    out["age_band_max_age"] = pd.to_numeric(out["age_band_max_age"], errors="raise").astype("int64")
    return out


def _evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    classes = list(model.classes_)
    probs = model.predict_proba(X)[:, classes.index(1)]
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    tp = int(((y == 1) & (preds == 1)).sum())
    fp = int(((y == 0) & (preds == 1)).sum())
    net_value = round(tp * (RETENTION_VALUE_USD - CONTACT_COST_USD) - fp * CONTACT_COST_USD, 2)
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "f2": float(fbeta_score(y, preds, beta=2, zero_division=0)),
        "net_value_usd": net_value,
    }


def _load_window_data(batch_count: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ventana: histórico development + lotes etiquetados 1..(N-1); evalúa en lote N."""
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    gold_schema = _identifier("SNOWFLAKE_GOLD_SCHEMA", os.getenv("SNOWFLAKE_GOLD_SCHEMA", "GOLD"))
    obt_table = _identifier("SNOWFLAKE_OBT_TABLE", os.getenv("SNOWFLAKE_OBT_TABLE", "OBT_CUSTOMER_CHURN"))

    conn = _snowflake_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {", ".join(c.upper() for c in MODEL_INPUTS)}, EXITED
                FROM {database}.{gold_schema}.{obt_table}
                WHERE SOURCE_SPLIT = 'train'
                ORDER BY OBSERVATION_ID
                """
            )
            cols = [d[0].lower() for d in cursor.description]
            full = pd.DataFrame(cursor.fetchall(), columns=cols)
    finally:
        conn.close()

    full["exited"] = pd.to_numeric(full["exited"], errors="raise").astype(int)
    dev, holdout = train_test_split(
        full, test_size=0.20, random_state=42, stratify=full["exited"]
    )
    holdout_reset = holdout.reset_index(drop=True)
    splits = np.array_split(np.arange(len(holdout_reset)), batch_count)
    batches = [holdout_reset.iloc[idx].copy() for idx in splits]
    # Ventana de entrenamiento: histórico + lotes etiquetados 1..7 (lote 8 reservado)
    train_window = pd.concat([dev.reset_index(drop=True), *batches[:-1]], ignore_index=True)
    eval_batch = batches[-1].reset_index(drop=True)
    return train_window, eval_batch


def _record_retraining(record: dict[str, Any]) -> None:
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    schema = _identifier(
        "SNOWFLAKE_MONITORING_SCHEMA",
        os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
    )
    conn = _snowflake_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {database}.{schema}.REENTRENAMIENTOS (
                    RETRAINING_ID, CREATED_AT, TRIGGER_BATCH_ID, TRIGGER_REASON,
                    STATUS, TARGET_AVAILABLE, CANDIDATE_RUN_ID, EVALUATION_METRIC,
                    PROMOTION, DECISION, REASON
                )
                SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s, %s, %s, %s, %s
                """,
                (
                    record["retraining_id"],
                    record["trigger_batch_id"],
                    record["trigger_reason"],
                    record["status"],
                    record["target_available"],
                    record["candidate_run_id"],
                    record["evaluation_metric"],
                    record["promotion"],
                    record["decision"],
                    record["reason"],
                ),
            )
    finally:
        conn.close()


def train_and_compare(
    trigger_batch_id: str,
    trigger_reason: str,
    tracking_uri: str,
    output_file: Path | None = None,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    train_window, eval_batch = _load_window_data(batch_count=8)

    X_train = _prepare_model_frame(train_window)
    y_train = train_window["exited"]
    X_eval = _prepare_model_frame(eval_batch)
    y_eval = eval_batch["exited"]

    # Reentrenamiento con el mismo pipeline y los hiperparámetros congelados del champion
    pipeline = make_model_pipeline(GradientBoostingClassifier(**CHAMPION_HYPERPARAMS))
    pipeline.fit(X_train, y_train)

    champion_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    champion_model = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{champion_mv.version}"
    )

    champ_metrics = _evaluate_model(champion_model, X_eval, y_eval)
    chall_metrics = _evaluate_model(pipeline, X_eval, y_eval)

    # Regla de promoción (Sección 4): mejora negocio Y supera ROC-AUC en > 1 SD de CV
    meets_rule = (
        chall_metrics["net_value_usd"] > champ_metrics["net_value_usd"]
        and chall_metrics["roc_auc"] > champ_metrics["roc_auc"] + CV_ROC_AUC_STD
    )

    experiment = mlflow.set_experiment("pset4-churn-retraining")
    input_example = X_train.head(5).copy()
    signature = infer_signature(input_example, pipeline.predict(input_example))

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"retrain_{trigger_batch_id}",
    ) as run:
        mlflow.set_tags(
            {
                "fase": "reentrenamiento",
                "trigger_batch_id": trigger_batch_id,
                "trigger_reason": trigger_reason,
                "window_policy": "historico_mas_lotes_1_a_7_eval_lote_8",
            }
        )
        mlflow.log_params(
            {
                "fase": "reentrenamiento",
                "model_family": "gradient_boosting",
                "train_rows": len(X_train),
                "eval_rows": len(X_eval),
                "champion_version_compared": str(champion_mv.version),
                **CHAMPION_HYPERPARAMS,
            }
        )
        mlflow.log_metrics(
            {
                "challenger_roc_auc": chall_metrics["roc_auc"],
                "challenger_average_precision": chall_metrics["average_precision"],
                "challenger_f2": chall_metrics["f2"],
                "challenger_net_value_usd": chall_metrics["net_value_usd"],
                "champion_roc_auc": champ_metrics["roc_auc"],
                "champion_net_value_usd": champ_metrics["net_value_usd"],
                "delta_roc_auc": chall_metrics["roc_auc"] - champ_metrics["roc_auc"],
                "delta_net_value_usd": chall_metrics["net_value_usd"] - champ_metrics["net_value_usd"],
                "meets_promotion_rule": float(meets_rule),
            }
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
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

    registered = next(
        (
            mv
            for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if mv.run_id == run_id
        ),
        None,
    )
    if registered is None:
        raise RuntimeError("No se pudo recuperar la versión challenger registrada.")

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, registered.version
    )

    summary = {
        "trigger_batch_id": trigger_batch_id,
        "trigger_reason": trigger_reason,
        "champion_version": str(champion_mv.version),
        "challenger_version": str(registered.version),
        "challenger_run_id": run_id,
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics,
        "meets_promotion_rule": meets_rule,
        "recommendation": "aprobar" if meets_rule else "rechazar",
    }

    _record_retraining(
        {
            "retraining_id": f"{trigger_batch_id}:v{registered.version}:pending",
            "trigger_batch_id": trigger_batch_id,
            "trigger_reason": trigger_reason,
            "status": "awaiting_human_approval",
            "target_available": True,
            "candidate_run_id": run_id,
            "evaluation_metric": chall_metrics["roc_auc"],
            "promotion": "recommended" if meets_rule else "not_recommended",
            "decision": "pending_approval",
            "reason": (
                f"Challenger v{registered.version} (AUC={chall_metrics['roc_auc']:.4f}, "
                f"Net=${chall_metrics['net_value_usd']:.2f}) vs Champion v{champion_mv.version} "
                f"(AUC={champ_metrics['roc_auc']:.4f}, Net=${champ_metrics['net_value_usd']:.2f})."
            ),
        }
    )

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def apply_human_decision(
    trigger_batch_id: str,
    challenger_version: str,
    challenger_run_id: str,
    approve: bool,
    reason: str,
    tracking_uri: str,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if approve:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, challenger_version
        )
        decision = "promoted_to_champion"
        promotion = "promoted"
        status = "completed"
    else:
        decision = "rejected_keep_champion"
        promotion = "rejected"
        status = "completed"

    record = {
        "retraining_id": f"{trigger_batch_id}:v{challenger_version}:{promotion}",
        "trigger_batch_id": trigger_batch_id,
        "trigger_reason": "human_approval_gate",
        "status": status,
        "target_available": True,
        "candidate_run_id": challenger_run_id,
        "evaluation_metric": None,
        "promotion": promotion,
        "decision": decision,
        "reason": reason,
    }
    _record_retraining(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--batch-id", required=True)
    p_train.add_argument("--trigger-reason", default="concept_drift_2_batches")
    p_train.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001"),
    )
    p_train.add_argument("--output-file", type=Path, default=None)

    p_decide = sub.add_parser("decide")
    p_decide.add_argument("--batch-id", required=True)
    p_decide.add_argument("--challenger-version", required=True)
    p_decide.add_argument("--challenger-run-id", required=True)
    p_decide.add_argument("--approve", required=True)
    p_decide.add_argument("--reason", default="Decisión registrada desde Kestra.")
    p_decide.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001"),
    )

    args = parser.parse_args()
    if args.command == "train":
        res = train_and_compare(
            args.batch_id, args.trigger_reason, args.tracking_uri, args.output_file
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        approve_flag = str(args.approve).strip().lower() in {"true", "1", "yes", "si"}
        res = apply_human_decision(
            args.batch_id,
            args.challenger_version,
            args.challenger_run_id,
            approve_flag,
            args.reason,
            args.tracking_uri,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()