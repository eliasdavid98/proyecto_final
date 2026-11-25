"""Vista operativa del servicio, deriva y alertas guardadas en Snowflake."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import pandas as pd
import snowflake.connector
import streamlit as st

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def _identifier(name: str, value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} debe ser un identificador Snowflake válido.")
    return value.upper()


def _snowflake_connection() -> Any:
    database = os.getenv("SNOWFLAKE_DATABASE", "").strip()
    required = {
        "SNOWFLAKE_ACCOUNT": os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
        "SNOWFLAKE_USERNAME": os.getenv("SNOWFLAKE_USERNAME", "").strip(),
        "SNOWFLAKE_PASSWORD": os.getenv("SNOWFLAKE_PASSWORD", "").strip(),
        "SNOWFLAKE_WAREHOUSE": os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
        "SNOWFLAKE_DATABASE": database,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Faltan variables Snowflake: {', '.join(missing)}.")
    options: dict[str, Any] = {
        "account": required["SNOWFLAKE_ACCOUNT"],
        "user": required["SNOWFLAKE_USERNAME"],
        "password": required["SNOWFLAKE_PASSWORD"],
        "database": _identifier("SNOWFLAKE_DATABASE", database),
        "warehouse": required["SNOWFLAKE_WAREHOUSE"],
        "login_timeout": 10,
    }
    role = os.getenv("SNOWFLAKE_ROLE", "").strip()
    if role:
        options["role"] = role
    return snowflake.connector.connect(**options)


def _query(sql: str, parameters: tuple[Any, ...] = ()) -> pd.DataFrame:
    connection = _snowflake_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            columns = [description[0].lower() for description in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        connection.close()


def _table(name: str) -> str:
    database = _identifier("SNOWFLAKE_DATABASE", os.getenv("SNOWFLAKE_DATABASE", ""))
    schema = _identifier(
        "SNOWFLAKE_MONITORING_SCHEMA",
        os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
    )
    return f"{database}.{schema}.{name}"


def _get_api(path: str) -> dict[str, Any]:
    base_url = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"API no disponible en {base_url}: {error}") from error


st.set_page_config(page_title="PSet 4 | Churn", layout="wide")
st.title("PSet 4 · Operación del modelo de churn")
st.caption(
    "Demo técnica con churn bancario. Los lotes y sus semanas son sintéticos; "
    "no prueban deriva temporal ni validan el caso telecom."
)

left, right = st.columns(2)
with left:
    try:
        model = _get_api("/modelo")
        st.metric("Estado del modelo", model["service_status"])
        st.json(model)
    except (RuntimeError, KeyError) as error:
        st.error(str(error))
with right:
    try:
        metrics = _get_api("/metricas/sistema")
        st.metric("Solicitudes API registradas", metrics["request_count"])
        st.json(metrics)
    except (RuntimeError, KeyError) as error:
        st.error(str(error))

st.subheader("Monitoreo por lote")
try:
    monitoring = _query(
        f"""
        SELECT BATCH_ID, CREATED_AT, SYNTHETIC_PERIOD_START, ROW_COUNT,
               INDUCED_DRIFT, MEAN_PROBABILITY, PREDICTED_CONTACT_RATE,
               TARGET_AVAILABLE, MAX_PSI, ALERT_COUNT, OBT_SNAPSHOT_ID,
               MODEL_ALIAS, MODEL_VERSION
        FROM {_table("MONITOREO")}
        ORDER BY CREATED_AT DESC
        LIMIT 100
        """
    )
    if monitoring.empty:
        st.info("Aún no hay lotes registrados.")
    else:
        st.dataframe(monitoring, use_container_width=True, hide_index=True)
except Exception as error:
    st.error(f"No se pudo consultar MONITOREO: {error}")

st.subheader("Alertas")
try:
    alerts = _query(
        f"""
        SELECT ALERT_ID, CREATED_AT, BATCH_ID, METRIC, VALUE, THRESHOLD,
               SEVERITY, RECIPIENT, ACTION
        FROM {_table("ALERTAS")}
        ORDER BY CREATED_AT DESC
        LIMIT 200
        """
    )
    if alerts.empty:
        st.info("Aún no hay alertas.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)
except Exception as error:
    st.error(f"No se pudo consultar ALERTAS: {error}")

st.subheader("Gobernanza de reentrenamiento")
try:
    retraining = _query(
        f"""
        SELECT RETRAINING_ID, CREATED_AT, TRIGGER_BATCH_ID, STATUS,
               TARGET_AVAILABLE, PROMOTION, DECISION, REASON
        FROM {_table("REENTRENAMIENTOS")}
        ORDER BY CREATED_AT DESC
        LIMIT 100
        """
    )
    if retraining.empty:
        st.info("Aún no hay decisiones de reentrenamiento.")
    else:
        st.dataframe(retraining, use_container_width=True, hide_index=True)
except Exception as error:
    st.error(f"No se pudo consultar REENTRENAMIENTOS: {error}")
