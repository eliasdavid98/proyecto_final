"""Runs an eight-slice synthetic batch demo against the Snowflake OBT and API."""

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

import pandas as pd
import snowflake.connector

from monitoreo.alertas import AlertThresholds, build_drift_alerts
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


def _load_reference(cursor: Any, source_table: str) -> pd.DataFrame:
    cursor.execute(
        f"""
        SELECT {", ".join(column.upper() for column in MODEL_INPUTS)}
        FROM {source_table}
        WHERE SOURCE_SPLIT = 'train'
        """
    )
    reference = _records(cursor)
    if reference.empty:
        raise ValueError("La partición train de la OBT no contiene filas.")
    return _canonicalize(reference)


def _load_snapshot_hash(cursor: Any, source_table: str) -> str:
    cursor.execute(f"SELECT HASH_AGG(*) FROM {source_table}")
    result = cursor.fetchone()
    if not result or result[0] is None:
        raise ValueError("No se pudo calcular HASH_AGG de la OBT.")
    return f"snowflake-hash-agg:{source_table}:{result[0]}"


def _load_test_batch(
    cursor: Any, source_table: str, batch_number: int, batch_count: int
) -> pd.DataFrame:
    cursor.execute(
        f"""
        WITH numbered AS (
            SELECT
                {", ".join(column.upper() for column in MODEL_INPUTS)},
                OBSERVATION_ID,
                ROW_NUMBER() OVER (ORDER BY OBSERVATION_ID) AS ROW_NUM,
                COUNT(*) OVER () AS TOTAL_ROWS
            FROM {source_table}
            WHERE SOURCE_SPLIT = 'test'
        )
        SELECT *
        FROM numbered
        WHERE ROW_NUM > FLOOR((%(batch_number)s - 1) * TOTAL_ROWS / %(batch_count)s)
          AND ROW_NUM <= FLOOR(%(batch_number)s * TOTAL_ROWS / %(batch_count)s)
        ORDER BY ROW_NUM
        """,
        {"batch_number": batch_number, "batch_count": batch_count},
    )
    batch = _records(cursor)
    if batch.empty:
        raise ValueError(
            f"La OBT no tiene filas test para el lote {batch_number}/{batch_count}."
        )
    batch["observation_id"] = batch["observation_id"].astype(str)
    return _canonicalize(batch)


def _induce_drift(batch: pd.DataFrame) -> pd.DataFrame:
    changed = batch.copy()
    changed["balance"] = changed["balance"] + 50000.0
    changed["geography"] = "Germany"
    changed["balance_status"] = "saldo_positivo"
    return changed


def _api_features(row: pd.Series) -> dict[str, Any]:
    result = {column: row[column] for column in MODEL_INPUTS}
    for column in NUMERIC_FEATURES:
        if column == "number_of_products":
            result[column] = int(result[column])
        elif column in ("has_credit_card", "is_active_member"):
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
            RUN_ID VARCHAR
        )
        """
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
            MODEL_NAME, MODEL_ALIAS, MODEL_VERSION, RUN_ID
        )
        SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s, %s, %s, %s, %s,
               %s, NULL, NULL, NULL, %s, %s, %s, 0, 0, %s, PARSE_JSON(%s),
               %s, FALSE, %s, %s, %s, %s, %s, %s
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
            values["api_request_count"],
            values["api_p50_latency_ms"],
            values["api_p95_latency_ms"],
            values["max_psi"],
            json.dumps(values["feature_metrics"], allow_nan=False),
            values["alert_count"],
            values["retraining_block_reason"],
            values["snapshot_id"],
            values["model_name"],
            values["model_alias"],
            values["model_version"],
            values["run_id"],
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


def _write_retraining_gate(
    cursor: Any,
    database: str,
    schema: str,
    batch_id: str,
    alert_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    high_psi = any(
        alert["severity"] == "alta" and alert["metric"].startswith("PSI(")
        for alert in alert_rows
    )
    record = {
        "retraining_id": f"{batch_id}:retraining-gate",
        "trigger_batch_id": batch_id,
        "trigger": "PSI > 0.25" if high_psi else "sin_alerta_critica",
        "status": "blocked",
        "target_available": False,
        "candidate_run_id": None,
        "evaluation_metric": None,
        "promotion": "not_evaluated",
        "decision": "no_retraining",
        "reason": (
            "El lote test de competencia no tiene etiquetas; se bloquea el "
            "reentrenamiento para no inventar targets ni reutilizar el holdout."
        ),
    }
    cursor.execute(
        f"""
        INSERT INTO {database}.{schema}.REENTRENAMIENTOS (
            RETRAINING_ID, CREATED_AT, TRIGGER_BATCH_ID, TRIGGER_REASON, STATUS,
            TARGET_AVAILABLE, CANDIDATE_RUN_ID, EVALUATION_METRIC,
            PROMOTION, DECISION, REASON
        )
        SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, FALSE, NULL, NULL,
               %s, %s, %s
        """,
        (
            record["retraining_id"],
            record["trigger_batch_id"],
            record["trigger"],
            record["status"],
            record["promotion"],
            record["decision"],
            record["reason"],
        ),
    )
    return record


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
    retraining_rows: list[dict[str, Any]] = []
    try:
        with connection.cursor() as cursor:
            reference = _load_reference(cursor, source_table)
            snapshot_id = _load_snapshot_hash(cursor, source_table)
            _create_monitoring_tables(cursor, database, monitoring_schema)
            for index in range(batch_count):
                batch_number = index + 1
                batch_id = f"{execution_id}-synthetic-week-{batch_number:02d}"
                batch = _load_test_batch(cursor, source_table, batch_number, batch_count)
                induced = batch_number == batch_count
                if induced:
                    batch = _induce_drift(batch)
                feature_metrics = compare_feature_distributions(
                    reference,
                    batch,
                    numeric_features=NUMERIC_FEATURES,
                    categorical_features=CATEGORICAL_FEATURES,
                )
                alerts = build_drift_alerts(
                    batch_id, feature_metrics, AlertThresholds()
                )
                responses, api_latency_ms = _call_batch_api(
                    api_base_url, batch_id, snapshot_id, batch
                )
                probabilities = [float(item["probabilidad"]) for item in responses]
                decisions = [item["decision"] == "contactar" for item in responses]
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
                    "simulation_basis": "fila_de_archivo_sin_orden_temporal",
                    "induced_drift": induced,
                    "induced_change": (
                        "balance += 50000; geography='Germany'" if induced else None
                    ),
                    "mean_probability": statistics.fmean(probabilities),
                    "predicted_contact_rate": statistics.fmean(decisions),
                    "target_available": False,
                    "api_request_count": 1,
                    "api_p50_latency_ms": api_latency_ms,
                    "api_p95_latency_ms": api_latency_ms,
                    "max_psi": max(float(item["psi"]) for item in feature_metrics),
                    "feature_metrics": feature_metrics,
                    "alert_count": len(alerts),
                    "retraining_block_reason": (
                        "test sin etiquetas; requiere outcomes posteriores "
                        "y aprobación humana"
                    ),
                    "snapshot_id": snapshot_id,
                    "model_name": first_response["model_name"],
                    "model_alias": first_response["model_alias"],
                    "model_version": first_response["model_version"],
                    "run_id": first_response["run_id"],
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
                        "alert_ids": ",".join(
                            str(alert["alert_id"]) for alert in alerts
                        ),
                        "roc_auc": None,
                        "average_precision": None,
                        "f2_at_0_5": None,
                        "api_4xx_rate": None,
                        "api_5xx_rate": None,
                    }
                )
                alert_rows.extend(alerts)
                for response in responses:
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
                            "target_available": False,
                            "induced_drift": induced,
                        }
                    )

            retraining_rows.append(
                _write_retraining_gate(
                    cursor,
                    database,
                    monitoring_schema,
                    monitoring_rows[-1]["batch_id"],
                    [
                        alert
                        for alert in alert_rows
                        if alert["batch_id"] == monitoring_rows[-1]["batch_id"]
                    ],
                )
            )
    finally:
        connection.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predicciones.csv", index=False
    )
    pd.DataFrame(monitoring_rows).to_csv(output_dir / "monitoreo.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(output_dir / "alertas.csv", index=False)
    pd.DataFrame(retraining_rows).to_csv(
        output_dir / "reentrenamientos.csv", index=False
    )
    return {
        "prediction_rows": len(prediction_rows),
        "batch_count": len(monitoring_rows),
        "alerts": len(alert_rows),
        "retraining_rows": len(retraining_rows),
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
