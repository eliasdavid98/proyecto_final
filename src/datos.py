"""Carga de la OBT de Snowflake para los notebooks del PSet 3."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Falta la variable {name}. Configúrala en el archivo .env de PSet3."
        )
    return value


def _snowflake_identifier(name: str, value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{name} debe ser un identificador Snowflake sin comillas; "
            "se permiten letras, números, '_' y '$'."
        )
    return f'"{value}"'


def load_obt() -> pd.DataFrame:
    """Lee la OBT configurada en Snowflake y devuelve nombres de columnas en minúscula."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    account = _required_env("SNOWFLAKE_ACCOUNT")
    user = _required_env("SNOWFLAKE_USERNAME")
    password = _required_env("SNOWFLAKE_PASSWORD")
    warehouse = _required_env("SNOWFLAKE_WAREHOUSE")
    database = _snowflake_identifier(
        "SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE")
    )
    schema = _snowflake_identifier(
        "SNOWFLAKE_GOLD_SCHEMA",
        os.getenv("SNOWFLAKE_GOLD_SCHEMA", "GOLD").strip() or "GOLD",
    )
    table = _snowflake_identifier(
        "SNOWFLAKE_OBT_TABLE",
        os.getenv("SNOWFLAKE_OBT_TABLE", "OBT_CUSTOMER_CHURN").strip()
        or "OBT_CUSTOMER_CHURN",
    )

    connection_options: dict[str, str] = {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "database": database.strip('"'),
        "schema": schema.strip('"'),
    }
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        connection_options["role"] = role

    query = f"SELECT * FROM {database}.{schema}.{table}"
    connection = snowflake.connector.connect(**connection_options)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            columns = [description[0].lower() for description in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()

    frame = pd.DataFrame.from_records(rows, columns=columns)
    if frame.empty:
        raise ValueError(
            f"La tabla {database}.{schema}.{table} no contiene observaciones."
        )
    return frame
