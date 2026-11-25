"""Records the labeled-outcome and human-approval gate for retraining."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import snowflake.connector

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")
BATCH_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable requerida {name}.")
    return value


def _identifier(name: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} debe ser un identificador Snowflake válido.")
    return value.upper()


def record_gate(batch_id: str) -> dict[str, Any]:
    if not BATCH_PATTERN.fullmatch(batch_id):
        raise ValueError("batch-id solo acepta letras, números, guion y guion bajo.")
    database = _identifier("SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE"))
    schema = _identifier(
        "SNOWFLAKE_MONITORING_SCHEMA",
        os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
    )
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
    connection = snowflake.connector.connect(**options)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT_IF(TARGET_AVAILABLE)
                FROM {database}.{schema}.MONITOREO
                WHERE BATCH_ID = %s
                """,
                (batch_id,),
            )
            result = cursor.fetchone()
            labeled_rows = int(result[0] or 0) if result else 0
            target_available = labeled_rows > 0
            record = {
                "retraining_id": f"{batch_id}:retraining-gate",
                "trigger_batch_id": batch_id,
                "trigger": "manual_review",
                "status": (
                    "awaiting_human_approval" if target_available else "blocked"
                ),
                "target_available": target_available,
                "candidate_run_id": None,
                "evaluation_metric": None,
                "promotion": "not_evaluated",
                "decision": "no_retraining",
                "reason": (
                    "Hay outcomes etiquetados, pero la aprobación humana y la "
                    "evaluación temporal independiente siguen pendientes."
                    if target_available
                    else (
                        "No hay outcomes etiquetados para el lote. Se bloquea el "
                        "reentrenamiento para no inventar targets ni reutilizar "
                        "el holdout ya consumido."
                    )
                ),
            }
            cursor.execute(
                f"""
                INSERT INTO {database}.{schema}.REENTRENAMIENTOS (
                    RETRAINING_ID, CREATED_AT, TRIGGER_BATCH_ID, TRIGGER_REASON,
                    STATUS, TARGET_AVAILABLE, CANDIDATE_RUN_ID, EVALUATION_METRIC,
                    PROMOTION, DECISION, REASON
                )
                SELECT %s, CURRENT_TIMESTAMP(), %s, %s, %s, %s, NULL, NULL,
                       %s, %s, %s
                """,
                (
                    record["retraining_id"],
                    record["trigger_batch_id"],
                    record["trigger"],
                    record["status"],
                    target_available,
                    record["promotion"],
                    record["decision"],
                    record["reason"],
                ),
            )
    finally:
        connection.close()
    return record | {"labeled_rows": labeled_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    print(json.dumps(record_gate(args.batch_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
