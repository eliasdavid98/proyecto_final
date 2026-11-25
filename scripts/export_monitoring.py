"""Exports operational Snowflake tables without customer IDs or raw features."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import snowflake.connector

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")
EXPORTS = {
    "predicciones.csv": (
        "PREDICCIONES",
        """
        PREDICTION_ID, CREATED_AT, PROBABILITY, DECISION, ACTION, MODEL_NAME,
        MODEL_ALIAS, MODEL_VERSION, RUN_ID, FALLBACK_REASON, LATENCY_MS, USED_FALLBACK,
        BATCH_ID, OBT_SNAPSHOT_ID, SOURCE_GOLD_TABLES
        """,
    ),
    "monitoreo.csv": (
        "MONITOREO",
        """
        MONITORING_ID, CREATED_AT, BATCH_ID, SYNTHETIC_PERIOD_START, ROW_COUNT,
        SIMULATION_BASIS, INDUCED_DRIFT, INDUCED_CHANGE, MEAN_PROBABILITY,
        PREDICTED_CONTACT_RATE, TARGET_AVAILABLE, ROC_AUC, AVERAGE_PRECISION,
        F2_AT_0_5, API_REQUEST_COUNT, API_P50_LATENCY_MS, API_P95_LATENCY_MS,
        API_4XX_RATE, API_5XX_RATE, MAX_PSI, FEATURE_METRICS, ALERT_COUNT,
        RETRAINING_TRIGGERED, RETRAINING_BLOCK_REASON, OBT_SNAPSHOT_ID,
        MODEL_NAME, MODEL_ALIAS, MODEL_VERSION, RUN_ID
        """,
    ),
    "alertas.csv": (
        "ALERTAS",
        """
        ALERT_ID, CREATED_AT, BATCH_ID, METRIC, VALUE, THRESHOLD, WINDOW_LABEL,
        SEVERITY, RECIPIENT, ACTION, SUPPORTING_TEST, SUPPORTING_P_VALUE
        """,
    ),
    "reentrenamientos_operativos.csv": (
        "REENTRENAMIENTOS",
        """
        RETRAINING_ID, CREATED_AT, TRIGGER_BATCH_ID, TRIGGER_REASON, STATUS,
        TARGET_AVAILABLE, CANDIDATE_RUN_ID, EVALUATION_METRIC, PROMOTION,
        DECISION, REASON
        """,
    ),
}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable requerida {name}.")
    return value


def _identifier(name: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} debe ser un identificador Snowflake válido.")
    return value.upper()


def _connection() -> Any:
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    options: dict[str, Any] = {
        "account": _required_env("SNOWFLAKE_ACCOUNT"),
        "user": _required_env("SNOWFLAKE_USERNAME"),
        "password": _required_env("SNOWFLAKE_PASSWORD"),
        "database": database,
        "warehouse": _required_env("SNOWFLAKE_WAREHOUSE"),
        "login_timeout": 10,
    }
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        options["role"] = role
    return snowflake.connector.connect(**options)


def export(output_dir: Path) -> dict[str, int]:
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    schema = _identifier(
        "SNOWFLAKE_MONITORING_SCHEMA",
        os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    connection = _connection()
    try:
        with connection.cursor() as cursor:
            for filename, (table, columns) in EXPORTS.items():
                cursor.execute(
                    f"""
                    SELECT {columns}
                    FROM {database}.{schema}.{table}
                    ORDER BY CREATED_AT
                    """
                )
                names = [item[0].lower() for item in cursor.description]
                frame = pd.DataFrame(cursor.fetchall(), columns=names)
                frame.to_csv(output_dir / filename, index=False)
                row_counts[filename] = len(frame)
    finally:
        connection.close()
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    print(export(args.output_dir))


if __name__ == "__main__":
    main()
