"""Contratos HTTP para la inferencia del modelo bancario de churn."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AGE_BANDS = {
    "18-29": (18, 29),
    "30-39": (30, 39),
    "40-49": (40, 49),
    "50+": (50, 100),
}


class ModelFeatures(BaseModel):
    """Campos requeridos por la firma del modelo registrado en MLflow."""

    model_config = ConfigDict(extra="forbid")

    geography: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]
    age_band: Literal["18-29", "30-39", "40-49", "50+"]
    age_band_min_age: int = Field(ge=18, le=50)
    age_band_max_age: int = Field(ge=29, le=100)
    number_of_products: int = Field(ge=1, le=4)
    credit_score: float = Field(ge=300, le=850)
    age: float = Field(ge=18, le=100)
    tenure: float = Field(ge=0, le=10)
    balance: float = Field(ge=0)
    has_credit_card: Literal[0, 1]
    is_active_member: Literal[0, 1]
    estimated_salary: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_age_band(self) -> ModelFeatures:
        expected_minimum, expected_maximum = AGE_BANDS[self.age_band]
        if (
            self.age_band_min_age != expected_minimum
            or self.age_band_max_age != expected_maximum
        ):
            raise ValueError("age_band_min_age/max_age no coinciden con age_band.")
        if not expected_minimum <= self.age <= expected_maximum:
            raise ValueError("age no pertenece al rango indicado por age_band.")
        return self


class PredictionRequest(BaseModel):
    """Entradas para una observación; el identificador no es una feature."""

    model_config = ConfigDict(extra="forbid")

    features: ModelFeatures
    observation_id: str | None = Field(default=None, min_length=1, max_length=128)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, max_length=128)
    obt_snapshot_id: str | None = Field(default=None, max_length=256)
    predictions: list[PredictionRequest] = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    prediction_id: str
    observation_id: str | None
    probabilidad: float
    decision: Literal["contactar", "sin_contacto", "revision_humana"]
    accion: Literal["revision_humana_retencion", "sin_accion"]
    model_name: str
    model_alias: str
    model_version: str | None
    run_id: str | None
    used_fallback: bool
    fallback_reason: str | None
    latency_ms: float


class ModelInfo(BaseModel):
    service_status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    alias: str
    version: str | None
    run_id: str | None
    error: str | None
