"""Runs an eight-slice labeled batch demo against the Snowflake OBT and API."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import snowflake.connector
from sklearn.metrics import average_precision_score, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split

from monitoreo.alertas import (
    AlertThresholds,
    build_concept_drift_alerts,
    build_drift_alerts,
)
from monitoreo.deriva import compare_feature_distributions

NUMERIC_FEATURES = (
    "credit_score",
    "age",
    "tenure",
    "balance",
    "number_of_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
)
CATEGORICAL_FEATURES = ("geography", "gender", "age_band", "balance_status")
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
DEFAULT_START = date(2026, 1, 5)
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")
CONTACT_COST_USD = float(os.getenv("CONTACT_COST_USD", "25.0"))
RETENTION_VALUE_USD = float(os.getenv("RETENTION_VALUE_USD", "300.0"))


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


def _source_table() -> str:
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    schema = _identifier(
        "SNOWFLAKE_GOLD_SCHEMA", os.getenv("SNOWFLAKE_GOLD_SCHEMA", "GOLD")
    )
    table = _identifier(
        "SNOWFLAKE_OBT_TABLE",
        os.getenv("SNOWFLAKE_OBT_TABLE", "OBT_CUSTOMER_CHURN"),
    )
    return f"{database}.{schema}.{table}"


def _records(cursor: Any) -> pd.DataFrame:
    columns = [description[0].lower() for description in cursor.description]
    return pd.DataFrame(cursor.fetchall(), columns=columns)


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["balance_status"] = result["balance"].map(
        lambda value: (
            "saldo_cero"
            if value == 0
            else "negativo_invalido"
            if value < 0
            else "saldo_positivo"
        )
    )
    return result


def _load_labeled_splits(
    cursor: Any, source_table: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cursor.execute(
        f"""
        SELECT {", ".join(column.upper() for column in MODEL_INPUTS)},
               OBSERVATION_ID, EXITED
        FROM {source_table}
        WHERE SOURCE_SPLIT = 'train'
        ORDER BY OBSERVATION_ID
        """
    )
    full = _records(cursor)
    if full.empty:
        raise ValueError("La partición train de la OBT no contiene filas.")
    full["observation_id"] = full["observation_id"].astype(str)
    full["exited"] = pd.to_numeric(full["exited"], errors="raise").astype(int)
    full = _canonicalize(full)
    reference, holdout = train_test_split(
        full,
        test_size=0.20,
        random_state=42,
        stratify=full["exited"],
    )
    return reference.reset_index(drop=True), holdout.reset_index(drop=True)


def _load_snapshot_hash(cursor: Any, source_table: str) -> str:
    cursor.execute(f"SELECT HASH_AGG(*) FROM {source_table}")
    result = cursor.fetchone()
    if not result or result[0] is None:
        raise ValueError("No se pudo calcular HASH_AGG de la OBT.")
    return f"snowflake-hash-agg:{source_table}:{result[0]}"


def _induce_drift(batch: pd.DataFrame) -> pd.DataFrame:
    changed = batch.copy()
    changed["balance"] = changed["balance"] + 50000.0
    changed["geography"] = "Germany"
    changed["balance_status"] = "saldo_positivo"
    mask = changed.index % 3 == 0
    changed.loc[mask, "is_active_member"] = 1 - changed.loc[mask, "is_active_member"]
    changed.loc[mask, "exited"] = 1 - changed.loc[mask, "exited"]
    return changed


def _business_value(y_true: list[int], contacted: list[bool]) -> float:
    tp = sum(1 for y, c in zip(y_true, contacted) if y == 1 and c)
    fp = sum(1 for y, c in zip(y_true, contacted) if y == 0 and c)
    return round(tp * (RETENTION_VALUE_USD - CONTACT_COST_USD) - fp * CONTACT_COST_USD, 2)


def _business_rule_decisions(batch: pd.DataFrame) -> list[bool]:
    return [
        bool(
            int(row["number_of_products"]) >= 3
            or (float(row["age"]) >= 40 and int(row["is_active_member"]) == 0)
        )
        for _, row in batch.iterrows()
    ]


def _api_features(row: pd.Series) -> dict[str, Any]:
    result = {column: row[column] for column in MODEL_INPUTS}
    for column in NUMERIC_FEATURES:
        if column in ("number_of_products", "has_credit_card", "is_active_member"):
            result[column] = int(result[column])
        else:
            result[column] = float(result[column])
    result["age_band_min_age"] = int(result["age_band_min_age"])
    result["age_band_max_age"] = int(result["age_band_max_age"])
    return result


def _call_batch_api(
    api_base_url: str,
    batch_id: str,
    snapshot_id: str,
    batch: pd.DataFrame,
) -> tuple[list[dict[str, Any]], float]:
    payload = {
        "batch_id": batch_id,
        "obt_snapshot_id": snapshot_id,
        "predictions": [
            {
                "observation_id": row["observation_id"],
                "features": _api_features(row),
            }
            for _, row in batch.iterrows()
        ],
    }
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base_url.rstrip('/')}/predecir/lote",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            results = json.loads(response.read())
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"API devolvió HTTP {error.code} para batch_id={batch_id}: "
            f"{response_body}"
        ) from error
    if len(results) != len(batch):
        raise RuntimeError(
            f"API devolvió {len(results)} predicciones para {len(batch)} filas."
        )
    return results, elapsed_ms


def _create_monitoring_tables(cursor: Any, database: str, schema: str) -> None:
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.MONITOREO (
            MONITORING_ID VARCHAR(36) NOT NULL,
            CREATED_AT TIMESTAMP_TZ NOT NULL,
            BATCH_ID VARCHAR NOT NULL,
            SYNTHETIC_PERIOD_START DATE NOT NULL,
            ROW_COUNT NUMBER NOT NULL,
            SIMULATION_BASIS VARCHAR NOT NULL,
            INDUCED_DRIFT BOOLEAN NOT NULL,
            INDUCED_CHANGE VARCHAR,
            MEAN_PROBABILITY FLOAT,
            PREDICTED_CONTACT_RATE FLOAT,
            TARGET_AVAILABLE BOOLEAN NOT NULL,
            ROC_AUC FLOAT,
            AVERAGE_PRECISION FLOAT,
            F2_AT_0_5 FLOAT,
            API_REQUEST_COUNT NUMBER,
            API_P50_LATENCY_MS FLOAT,
            API_P95_LATENCY_MS FLOAT,
            API_4XX_RATE FLOAT,
            API_5XX_RATE FLOAT,
            MAX_PSI FLOAT NOT NULL,
            FEATURE_METRICS VARIANT NOT NULL,
            ALERT_COUNT NUMBER NOT NULL,
            RETRAINING_TRIGGERED BOOLEAN NOT NULL,
            RETRAINING_BLOCK_REASON VARCHAR,
            OBT_SNAPSHOT_ID VARCHAR NOT NULL,
            MODEL_NAME VARCHAR,
            MODEL_ALIAS VARCHAR,
            MODEL_VERSION VARCHAR,
            RUN_ID VARCHAR,
            MODEL_NET_VALUE_USD FLOAT,
            BASELINE_NET_VALUE_USD FLOAT
        )
        """
    )
    cursor.execute(
        f"ALTER TABLE {database}.{schema}.MONITOREO ADD COLUMN IF NOT EXISTS MODEL_NET_VALUE_USD FLOAT"
    )
    cursor.execute(
        f"ALTER TABLE {database}.{schema}.MONITOREO ADD COLUMN IF NOT EXISTS BASELINE_NET_VALUE_USD FLOAT"
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.ALERTAS (
            ALERT_ID VARCHAR(256) NOT NULL,
            CREATED_AT TIMESTAMP_TZ NOT NULL,
            BATCH_ID VARCHAR NOT NULL,
            METRIC VARCHAR NOT NULL,
            VALUE FLOAT NOT NULL,
            THRESHOLD FLOAT NOT NULL,
            WINDOW_LABEL VARCHAR NOT NULL,
            SEVERITY VARCHAR NOT NULL,
            RECIPIENT VARCHAR NOT NULL,
            ACTION VARCHAR NOT NULL,
            SUPPORTING_TEST VARCHAR,
            SUPPORTING_P_VALUE FLOAT
        )
        """
    )
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.REENTRENAMIENTOS (
            RETRAINING_ID VARCHAR(256) NOT NULL,
            CREATED_AT TIMESTAMP_TZ NOT NULL,
            TRIGGER_BATCH_ID VARCHAR NOT NULL,
            TRIGGER_REASON VARCHAR NOT NULL,
            STATUS VARCHAR NOT NULL,
            TARGET_AVAILABLE BOOLEAN NOT NULL,
            CANDIDATE_RUN_ID VARCHAR,
            EVALUATION_METRIC FLOAT,
            PROMOTION VARCHAR NOT NULL,
            DECISION VARCHAR NOT NULL,
            REASON VARCHAR NOT NULL
        )
        """
    )


def _persist_monitoring(
    cursor: Any,
    database: str,
    schema: str,
    values: dict[str, Any],
    alerts: list[dict[str, Any]],
) -> None:
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.MONITOREO (
            MONITORING_ID, CREATED_AT, BATCH_ID, SYNTHETIC_PERIOD_START,
            ROW_COUNT, SIMULATION_BASIS, INDUCED_DRIFT, INDUCED_CHANGE,
            MEAN_PROBABILITY, PREDICTED_CONTACT_RATE, TARGET_AVAILABLE,
            ROC_AUC, AVERAGE_PRECISION, F2_AT_0_5, API_REQUEST_COUNT,
            API_P50_LATENCY_MS, API_P95_LATENCY_MS, API_4XX_RATE,
            API_5XX_RATE, MAX_PSI, FEATURE_METRICS, ALERT_COUNT,
            RETRAINING_TRIGGERED, RETRAINING_BLOCK_REASON, OBT_SNAPSHOT_ID,
            MODEL_NAME, MODEL_ALIAS, MODEL_VERSION, RUN_ID,
            MODEL_NET_VALUE_USD, BASELINE_NET_VALUE_USD
        )
        SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s, %s, %s, %s,
               %s, %s, %s, %s, %s, %s, %s, 0, 0, %s, PARSE_JSON(%s),
               %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        """,
        (
            values["monitoring_id"],
            values["batch_id"],
            values["period_start"],
            values["rows"],
            values["simulation_basis"],
            values["induced_drift"],
            values["induced_change"],
            values["mean_probability"],
            values["predicted_contact_rate"],
            values["target_available"],
            values["roc_auc"],
            values["average_precision"],
            values["f2_at_0_5"],
            values["api_request_count"],
            values["api_p50_latency_ms"],
            values["api_p95_latency_ms"],
            values["max_psi"],
            json.dumps(values["feature_metrics"], allow_nan=False),
            values["alert_count"],
            values["retraining_triggered"],
            values["retraining_block_reason"],
            values["snapshot_id"],
            values["model_name"],
            values["model_alias"],
            values["model_version"],
            values["run_id"],
            values["model_net_value_usd"],
            values["baseline_net_value_usd"],
        ),
    )
    for alert in alerts:
        cursor.execute(
            f"""
            INSERT INTO {database}.{schema}.ALERTAS (
                ALERT_ID, CREATED_AT, BATCH_ID, METRIC, VALUE, THRESHOLD,
                WINDOW_LABEL, SEVERITY, RECIPIENT, ACTION, SUPPORTING_TEST,
                SUPPORTING_P_VALUE
            )
            SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s
            """,
            (
                alert["alert_id"],
                alert["batch_id"],
                alert["metric"],
                alert["value"],
                alert["threshold"],
                alert["window"],
                alert["severity"],
                alert["recipient"],
                alert["action"],
                alert["supporting_test"],
                alert["supporting_p_value"],
            ),
        )


def run(
    api_base_url: str,
    batch_count: int,
    execution_id: str,
    output_dir: Path,
    start_date: date = DEFAULT_START,
) -> dict[str, int]:
    if batch_count < 8:
        raise ValueError("El demo del PSet 4 requiere al menos ocho lotes.")
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    monitoring_schema = _identifier(
        "SNOWFLAKE_MONITORING_SCHEMA",
        os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
    )
    source_table = _source_table()
    connection = _snowflake_connection()
    prediction_rows: list[dict[str, Any]] = []
    monitoring_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    consecutive_concept_alerts = 0
    thresholds = AlertThresholds()

    try:
        with connection.cursor() as cursor:
            reference, holdout = _load_labeled_splits(cursor, source_table)
            snapshot_id = _load_snapshot_hash(cursor, source_table)
            _create_monitoring_tables(cursor, database, monitoring_schema)
            splits = np.array_split(np.arange(len(holdout)), batch_count)

            for index, row_indices in enumerate(splits):
                batch_number = index + 1
                batch_id = f"{execution_id}-synthetic-week-{batch_number:02d}"
                batch = holdout.iloc[row_indices].copy().reset_index(drop=True)
                induced = batch_number in (6, 7)
                if induced:
                    batch = _induce_drift(batch)

                feature_metrics = compare_feature_distributions(
                    reference,
                    batch,
                    numeric_features=NUMERIC_FEATURES,
                    categorical_features=CATEGORICAL_FEATURES,
                )
                responses, api_latency_ms = _call_batch_api(
                    api_base_url, batch_id, snapshot_id, batch
                )

                probabilities = [float(item["probabilidad"]) for item in responses]
                decisions = [item["decision"] == "contactar" for item in responses]
                y_true = batch["exited"].astype(int).tolist()

                roc_auc = float(roc_auc_score(y_true, probabilities))
                avg_precision = float(average_precision_score(y_true, probabilities))
                f2_val = float(fbeta_score(y_true, decisions, beta=2, zero_division=0))
                model_net_value = _business_value(y_true, decisions)
                rule_net_value = _business_value(y_true, _business_rule_decisions(batch))

                drift_alerts = build_drift_alerts(batch_id, feature_metrics, thresholds)
                concept_alerts = build_concept_drift_alerts(
                    batch_id, roc_auc, avg_precision, thresholds=thresholds
                )
                alerts = drift_alerts + concept_alerts

                if concept_alerts:
                    consecutive_concept_alerts += 1
                else:
                    consecutive_concept_alerts = 0

                retraining_triggered = (
                    consecutive_concept_alerts
                    >= thresholds.consecutive_performance_batches
                )

                period_start = start_date + timedelta(days=index * 7)
                feature_metric_json = json.dumps(
                    feature_metrics, allow_nan=False, ensure_ascii=False
                )
                first_response = responses[0]

                monitor = {
                    "monitoring_id": str(uuid.uuid4()),
                    "batch_id": batch_id,
                    "period_start": period_start.isoformat(),
                    "rows": len(batch),
                    "simulation_basis": "holdout_etiquetado_8_lotes_sinteticos",
                    "induced_drift": induced,
                    "induced_change": (
                        "balance += 50000; geography='Germany'; inversion 33% is_active/exited"
                        if induced
                        else None
                    ),
                    "mean_probability": statistics.fmean(probabilities),
                    "predicted_contact_rate": statistics.fmean(decisions),
                    "target_available": True,
                    "roc_auc": roc_auc,
                    "average_precision": avg_precision,
                    "f2_at_0_5": f2_val,
                    "api_request_count": 1,
                    "api_p50_latency_ms": api_latency_ms,
                    "api_p95_latency_ms": api_latency_ms,
                    "max_psi": max(float(item["psi"]) for item in feature_metrics),
                    "feature_metrics": feature_metrics,
                    "alert_count": len(alerts),
                    "retraining_triggered": retraining_triggered,
                    "retraining_block_reason": (
                        "Disparador cumplido (2 lotes consecutivos < CV - 2*SD); ejecutar pset4_reentrenamiento"
                        if retraining_triggered
                        else None
                    ),
                    "snapshot_id": snapshot_id,
                    "model_name": first_response["model_name"],
                    "model_alias": first_response["model_alias"],
                    "model_version": first_response["model_version"],
                    "run_id": first_response["run_id"],
                    "model_net_value_usd": model_net_value,
                    "baseline_net_value_usd": rule_net_value,
                }
                _persist_monitoring(
                    cursor, database, monitoring_schema, monitor, alerts
                )
                monitoring_rows.append(
                    {
                        key: value
                        for key, value in monitor.items()
                        if key not in {"monitoring_id", "feature_metrics"}
                    }
                    | {
                        "feature_metrics_json": feature_metric_json,
                        "alert_ids": ",".join(str(alert["alert_id"]) for alert in alerts),
                        "api_4xx_rate": 0.0,
                        "api_5xx_rate": 0.0,
                    }
                )
                alert_rows.extend(alerts)
                for response, target_val in zip(responses, y_true):
                    prediction_rows.append(
                        {
                            "prediction_id": response["prediction_id"],
                            "batch_id": batch_id,
                            "probability": response["probabilidad"],
                            "decision": response["decision"],
                            "action": response["accion"],
                            "model_name": response["model_name"],
                            "model_alias": response["model_alias"],
                            "model_version": response["model_version"],
                            "run_id": response["run_id"],
                            "used_fallback": response["used_fallback"],
                            "latency_ms": response["latency_ms"],
                            "obt_snapshot_id": snapshot_id,
                            "target_available": True,
                            "target": target_val,
                            "induced_drift": induced,
                        }
                    )
    finally:
        connection.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predicciones.csv", index=False)
    pd.DataFrame(monitoring_rows).to_csv(output_dir / "monitoreo.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(output_dir / "alertas.csv", index=False)
    return {
        "prediction_rows": len(prediction_rows),
        "batch_count": len(monitoring_rows),
        "alerts": len(alert_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL", "http://api:8000"))
    parser.add_argument("--batch-count", type=int, default=8)
    parser.add_argument("--execution-id", default=f"local-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", args.execution_id):
        raise ValueError("execution-id acepta solo letras, números, guion y guion bajo.")
    results = run(
        api_base_url=args.api_base_url,
        batch_count=args.batch_count,
        execution_id=args.execution_id,
        output_dir=args.output_dir,
        start_date=args.start_date,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()