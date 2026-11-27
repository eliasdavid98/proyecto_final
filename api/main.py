"""Servicio FastAPI con carga dinámica del alias champion y fallback explícito."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
import snowflake.connector
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from mlflow import MlflowClient

from api.esquemas import (
    BatchPredictionRequest,
    ModelFeatures,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from src.seleccion import CHAMPION_ALIAS, REGISTERED_MODEL_NAME

logger = logging.getLogger("pset4.api")

INPUT_COLUMNS = (
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
DUMMY_PRIOR_PROBABILITY = 0.20641666666666666
DEFAULT_REFRESH_SECONDS = 5.0
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")

def evaluate_business_rule(features: ModelFeatures) -> bool:
    """Línea base de negocio bancaria interpretable sin ML."""
    return bool(
        features.number_of_products >= 3
        or (features.age >= 40 and features.is_active_member == 0)
    )

@dataclass(frozen=True)
class ModelBundle:
    model: Any
    version: str
    run_id: str
    alias: str


@dataclass(frozen=True)
class InferenceResult:
    probability: float
    decision: Literal["contactar", "sin_contacto", "revision_humana"]
    action: Literal["revision_humana_retencion", "sin_accion"]
    model_version: str | None
    run_id: str | None
    used_fallback: bool
    fallback_reason: str | None


class RegistryModelProvider:
    """Refresca el alias sin reiniciar el contenedor y gestiona los respaldos."""

    def __init__(
        self,
        model_name: str = REGISTERED_MODEL_NAME,
        alias: str = CHAMPION_ALIAS,
        tracking_uri: str | None = None,
        refresh_interval: float = DEFAULT_REFRESH_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self.tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        )
        self.refresh_interval = refresh_interval
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        self._active: ModelBundle | None = None
        self._error: str | None = None
        self._last_refresh = 0.0
        self._lock = threading.RLock()

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_refresh < self.refresh_interval:
                return
            self._last_refresh = now
            try:
                version = self.client.get_model_version_by_alias(
                    self.model_name, self.alias
                )
                version_string = str(version.version)
                if self._active and self._active.version == version_string:
                    self._error = None
                    return

                model_uri = f"models:/{self.model_name}/{version_string}"
                model = mlflow.sklearn.load_model(model_uri)
                if not hasattr(model, "predict_proba"):
                    raise TypeError(
                        f"El modelo {model_uri} no expone predict_proba."
                    )
                self._active = ModelBundle(
                    model=model,
                    version=version_string,
                    run_id=version.run_id,
                    alias=self.alias,
                )
                self._error = None
                logger.info(
                    "Loaded MLflow model %s@%s version=%s run_id=%s",
                    self.model_name,
                    self.alias,
                    version_string,
                    version.run_id,
                )
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                logger.exception(
                    "Could not load MLflow alias %s@%s; fallback remains explicit",
                    self.model_name,
                    self.alias,
                )

    def model_info(self) -> ModelInfo:
        mode = _prediction_mode()
        forced_fallback = mode != "model"
        if not forced_fallback:
            self.refresh()
        model_loaded = self._active is not None and not forced_fallback
        return ModelInfo(
            service_status=(
                "ok" if model_loaded and self._error is None else "degraded"
            ),
            model_loaded=model_loaded,
            model_name=self.model_name,
            alias=self.alias,
            version=(
                self._active.version
                if self._active and not forced_fallback
                else None
            ),
            run_id=(
                self._active.run_id
                if self._active and not forced_fallback
                else None
            ),
            error=(
                f"PREDICTION_MODE={mode}"
                if forced_fallback
                else self._error
            ),
        )

    def infer(self, features: ModelFeatures) -> InferenceResult:
        mode = _prediction_mode()
        manual_review_all = mode == "manual_review_all"
        forced_fallback = mode != "model"
        if not forced_fallback:
            self.refresh()
        threshold = _decision_threshold()
        fallback_reason = (
            f"PREDICTION_MODE={mode}"
            if forced_fallback
            else self._error
        )
        if mode == "business_rule":
            rule_hit = evaluate_business_rule(features)
            return InferenceResult(
                probability=1.0 if rule_hit else 0.0,
                decision="contactar" if rule_hit else "sin_contacto",
                action="revision_humana_retencion" if rule_hit else "sin_accion",
                model_version=None,
                run_id=None,
                used_fallback=True,
                fallback_reason="PREDICTION_MODE=business_rule",
            )
        if self._active is None or forced_fallback:
            probability = DUMMY_PRIOR_PROBABILITY
            version = None
            run_id = None
            used_fallback = True
        else:
            row = features.model_dump()
            frame = pd.DataFrame([row], columns=INPUT_COLUMNS)
            for column in FLOAT_COLUMNS:
                frame[column] = pd.to_numeric(
                    frame[column], errors="raise"
                ).astype("float64")
            scores = self._active.model.predict_proba(frame)
            classes = list(self._active.model.classes_)
            if 1 not in classes:
                raise RuntimeError("El modelo cargado no contiene la clase positiva 1.")
            probability = float(scores[0, classes.index(1)])
            version = self._active.version
            run_id = self._active.run_id
            used_fallback = False

        positive = probability >= threshold
        return InferenceResult(
            probability=probability,
            decision=(
                "revision_humana"
                if manual_review_all
                else "contactar" if positive else "sin_contacto"
            ),
            action=(
                "revision_humana_retencion"
                if manual_review_all or positive
                else "sin_accion"
            ),
            model_version=version,
            run_id=run_id,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
        )


class SnowflakePredictionStore:
    """Persiste cada resultado en <database>.MONITORING.PREDICCIONES."""

    def write(self, record: dict[str, Any]) -> None:
        self.write_many([record])

    def write_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        account = _required_env("SNOWFLAKE_ACCOUNT")
        user = _required_env("SNOWFLAKE_USERNAME")
        password = _required_env("SNOWFLAKE_PASSWORD")
        database = _safe_identifier(
            "SNOWFLAKE_DATABASE", _required_env("SNOWFLAKE_DATABASE")
        )
        warehouse = _required_env("SNOWFLAKE_WAREHOUSE")
        schema = _safe_identifier(
            "SNOWFLAKE_MONITORING_SCHEMA",
            os.getenv("SNOWFLAKE_MONITORING_SCHEMA", "MONITORING"),
        )
        role = os.getenv("SNOWFLAKE_ROLE", "").strip()
        options: dict[str, str] = {
            "account": account,
            "user": user,
            "password": password,
            "database": database,
            "warehouse": warehouse,
            "login_timeout": "10",
        }
        if role:
            options["role"] = role

        table = f"{database}.{schema}.PREDICCIONES"
        connection = snowflake.connector.connect(**options)
        transaction_started = False
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        PREDICTION_ID VARCHAR(36) NOT NULL,
                        CREATED_AT TIMESTAMP_TZ NOT NULL,
                        OBSERVATION_ID VARCHAR(128),
                        INPUTS VARIANT NOT NULL,
                        PROBABILITY FLOAT NOT NULL,
                        DECISION VARCHAR NOT NULL,
                        ACTION VARCHAR NOT NULL,
                        MODEL_NAME VARCHAR NOT NULL,
                        MODEL_ALIAS VARCHAR NOT NULL,
                        MODEL_VERSION VARCHAR,
                        RUN_ID VARCHAR,
                        FALLBACK_REASON VARCHAR,
                        LATENCY_MS FLOAT NOT NULL,
                        USED_FALLBACK BOOLEAN NOT NULL,
                        BATCH_ID VARCHAR,
                        OBT_SNAPSHOT_ID VARCHAR,
                        SOURCE_GOLD_TABLES VARCHAR
                    )
                    """
                )
                cursor.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS FALLBACK_REASON VARCHAR
                    """
                )
                columns = (
                    "PREDICTION_ID, OBSERVATION_ID, INPUTS, PROBABILITY, DECISION, "
                    "ACTION, MODEL_NAME, MODEL_ALIAS, MODEL_VERSION, RUN_ID, "
                    "FALLBACK_REASON, LATENCY_MS, USED_FALLBACK, BATCH_ID, "
                    "OBT_SNAPSHOT_ID, SOURCE_GOLD_TABLES"
                )
                selected_columns = (
                    "source.PREDICTION_ID, CURRENT_TIMESTAMP(), "
                    "source.OBSERVATION_ID, PARSE_JSON(source.INPUTS), "
                    "source.PROBABILITY, source.DECISION, source.ACTION, "
                    "source.MODEL_NAME, source.MODEL_ALIAS, source.MODEL_VERSION, "
                    "source.RUN_ID, source.FALLBACK_REASON, source.LATENCY_MS, "
                    "source.USED_FALLBACK, source.BATCH_ID, source.OBT_SNAPSHOT_ID, "
                    "source.SOURCE_GOLD_TABLES"
                )
                row_placeholders = "(" + ", ".join(["%s"] * 16) + ")"
                row_values = [
                    (
                        record["prediction_id"],
                        record["observation_id"],
                        json.dumps(record["inputs"], ensure_ascii=False),
                        record["probability"],
                        record["decision"],
                        record["action"],
                        record["model_name"],
                        record["model_alias"],
                        record["model_version"],
                        record["run_id"],
                        record["fallback_reason"],
                        record["latency_ms"],
                        record["used_fallback"],
                        record["batch_id"],
                        record["obt_snapshot_id"],
                        record["source_gold_tables"],
                    )
                    for record in records
                ]
                cursor.execute("BEGIN")
                transaction_started = True
                for start in range(0, len(row_values), 500):
                    chunk = row_values[start : start + 500]
                    values_clause = ", ".join([row_placeholders] * len(chunk))
                    parameters = tuple(value for row in chunk for value in row)
                    cursor.execute(
                        f"""
                        INSERT INTO {table} (
                            PREDICTION_ID, CREATED_AT, OBSERVATION_ID, INPUTS,
                            PROBABILITY, DECISION, ACTION, MODEL_NAME, MODEL_ALIAS,
                            MODEL_VERSION, RUN_ID, FALLBACK_REASON, LATENCY_MS,
                            USED_FALLBACK, BATCH_ID, OBT_SNAPSHOT_ID, SOURCE_GOLD_TABLES
                        )
                        SELECT {selected_columns}
                        FROM VALUES {values_clause} AS source ({columns})
                        """,
                        parameters,
                    )
                cursor.execute("COMMIT")
                transaction_started = False
        except Exception:
            if transaction_started:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("ROLLBACK")
                except Exception:
                    logger.exception("Could not roll back prediction batch in Snowflake.")
            raise
        finally:
            connection.close()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable requerida {name}.")
    return value


def _safe_identifier(name: str, value: str) -> str:
    value = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{name} debe ser un identificador Snowflake válido.")
    return value.upper()


def _decision_threshold() -> float:
    raw = os.getenv("CHURN_DECISION_THRESHOLD", "0.5")
    try:
        threshold = float(raw)
    except ValueError as error:
        raise RuntimeError("CHURN_DECISION_THRESHOLD debe ser numérico.") from error
    if not 0 <= threshold <= 1:
        raise RuntimeError("CHURN_DECISION_THRESHOLD debe estar entre 0 y 1.")
    return threshold


def _prediction_mode() -> Literal[
    "model", "dummy_fallback", "manual_review_all", "business_rule"
]:
    mode = os.getenv("PREDICTION_MODE", "model").strip().lower()
    if mode in {"model", "dummy_fallback", "manual_review_all", "business_rule"}:
        return mode
    raise RuntimeError(
        "PREDICTION_MODE debe ser 'model', 'dummy_fallback', "
        "'manual_review_all' o 'business_rule'."
    )


def get_model_provider(request: Request) -> RegistryModelProvider:
    provider = getattr(request.app.state, "model_provider", None)
    if provider is None:
        provider = RegistryModelProvider()
        request.app.state.model_provider = provider
    return provider


def get_prediction_store(request: Request) -> SnowflakePredictionStore:
    store = getattr(request.app.state, "prediction_store", None)
    if store is None:
        store = SnowflakePredictionStore()
        request.app.state.prediction_store = store
    return store


app = FastAPI(
    title="PSet 4 — Churn prediction service",
    version="1.0.0",
    description=(
        "Servicio de demostración sobre churn bancario; no valida el caso telecom "
        "ni su regla comercial original."
    ),
)


@app.middleware("http")
async def collect_request_metrics(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        events = getattr(request.app.state, "request_metrics", None)
        if events is None:
            events = deque(maxlen=5000)
            request.app.state.request_metrics = events
        events.append(
            {
                "timestamp": time.time(),
                "path": request.url.path,
                "status_code": status_code,
                "latency_ms": latency_ms,
            }
        )


@app.get("/metricas/sistema")
def system_metrics(request: Request) -> dict[str, float | int]:
    events = list(getattr(request.app.state, "request_metrics", []))
    if not events:
        return {
            "request_count": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "error_4xx_rate": 0.0,
            "error_5xx_rate": 0.0,
        }
    latencies = sorted(float(event["latency_ms"]) for event in events)
    count = len(events)
    return {
        "request_count": count,
        "p50_latency_ms": float(pd.Series(latencies).quantile(0.50)),
        "p95_latency_ms": float(pd.Series(latencies).quantile(0.95)),
        "error_4xx_rate": sum(
            400 <= int(event["status_code"]) < 500 for event in events
        )
        / count,
        "error_5xx_rate": sum(
            500 <= int(event["status_code"]) < 600 for event in events
        )
        / count,
    }


@app.get("/salud", response_model=ModelInfo)
def health(
    provider: RegistryModelProvider = Depends(get_model_provider),
) -> ModelInfo:
    return provider.model_info()


@app.get("/modelo", response_model=ModelInfo)
def model_info(
    provider: RegistryModelProvider = Depends(get_model_provider),
) -> ModelInfo:
    if _prediction_mode() == "model":
        provider.refresh(force=True)
    return provider.model_info()


@app.post("/predecir", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    provider: RegistryModelProvider = Depends(get_model_provider),
    store: SnowflakePredictionStore = Depends(get_prediction_store),
    batch_id: str | None = Header(default=None, alias="X-Batch-ID"),
    snapshot_id: str | None = Header(default=None, alias="X-OBT-Snapshot-ID"),
) -> PredictionResponse:
    prediction_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        result = provider.infer(payload.features)
    except (RuntimeError, ValueError, TypeError) as error:
        logger.exception("Prediction failed for prediction_id=%s", prediction_id)
        raise HTTPException(status_code=503, detail=str(error)) from error
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    record = {
        "prediction_id": prediction_id,
        "observation_id": payload.observation_id,
        "inputs": payload.features.model_dump(),
        "probability": result.probability,
        "decision": result.decision,
        "action": result.action,
        "model_name": provider.model_name,
        "model_alias": provider.alias,
        "model_version": result.model_version,
        "run_id": result.run_id,
        "fallback_reason": result.fallback_reason,
        "latency_ms": latency_ms,
        "used_fallback": result.used_fallback,
        "batch_id": batch_id,
        "obt_snapshot_id": snapshot_id or os.getenv("OBT_SNAPSHOT_ID"),
        "source_gold_tables": os.getenv(
            "OBT_SOURCE_GOLD_TABLES",
            "FACT_CUSTOMER_CHURN,DIM_GEOGRAPHY,DIM_GENDER,"
            "DIM_AGE_BAND,DIM_PRODUCT_COUNT",
        ),
    }
    try:
        store.write(record)
    except Exception as error:
        logger.exception(
            "Could not persist prediction_id=%s to Snowflake", prediction_id
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Predicción {prediction_id} no se entrega porque no se pudo "
                "registrar en Snowflake."
            ),
        ) from error

    return PredictionResponse(
        prediction_id=prediction_id,
        observation_id=payload.observation_id,
        probabilidad=result.probability,
        decision=result.decision,
        accion=result.action,
        model_name=provider.model_name,
        model_alias=provider.alias,
        model_version=result.model_version,
        run_id=result.run_id,
        used_fallback=result.used_fallback,
        fallback_reason=result.fallback_reason,
        latency_ms=latency_ms,
    )


@app.post("/predecir/lote", response_model=list[PredictionResponse])
def predict_batch(
    payload: BatchPredictionRequest,
    provider: RegistryModelProvider = Depends(get_model_provider),
    store: SnowflakePredictionStore = Depends(get_prediction_store),
) -> list[PredictionResponse]:
    responses: list[PredictionResponse] = []
    records: list[dict[str, Any]] = []
    for prediction in payload.predictions:
        prediction_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            result = provider.infer(prediction.features)
        except (RuntimeError, ValueError, TypeError) as error:
            logger.exception(
                "Batch inference failed for prediction_id=%s", prediction_id
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        record = {
            "prediction_id": prediction_id,
            "observation_id": prediction.observation_id,
            "inputs": prediction.features.model_dump(),
            "probability": result.probability,
            "decision": result.decision,
            "action": result.action,
            "model_name": provider.model_name,
            "model_alias": provider.alias,
            "model_version": result.model_version,
            "run_id": result.run_id,
            "fallback_reason": result.fallback_reason,
            "latency_ms": latency_ms,
            "used_fallback": result.used_fallback,
            "batch_id": payload.batch_id,
            "obt_snapshot_id": payload.obt_snapshot_id
            or os.getenv("OBT_SNAPSHOT_ID"),
            "source_gold_tables": os.getenv(
                "OBT_SOURCE_GOLD_TABLES",
                "FACT_CUSTOMER_CHURN,DIM_GEOGRAPHY,DIM_GENDER,"
                "DIM_AGE_BAND,DIM_PRODUCT_COUNT",
            ),
        }
        records.append(record)
        responses.append(
            PredictionResponse(
                prediction_id=prediction_id,
                observation_id=prediction.observation_id,
                probabilidad=result.probability,
                decision=result.decision,
                accion=result.action,
                model_name=provider.model_name,
                model_alias=provider.alias,
                model_version=result.model_version,
                run_id=result.run_id,
                used_fallback=result.used_fallback,
                fallback_reason=result.fallback_reason,
                latency_ms=latency_ms,
            )
        )
    try:
        store.write_many(records)
    except Exception as error:
        logger.exception(
            "Could not persist batch_id=%s (%s predictions) to Snowflake",
            payload.batch_id,
            len(records),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"El lote {payload.batch_id} no se entrega porque no se pudo "
                "registrar en Snowflake."
            ),
        ) from error
    return responses
