from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import api.main as api_module
import numpy as np
from fastapi.testclient import TestClient

from api.main import (
    DUMMY_PRIOR_PROBABILITY,
    RegistryModelProvider,
    app,
    get_model_provider,
    get_prediction_store,
)


VALID_FEATURES = {
    "geography": "France",
    "gender": "Female",
    "age_band": "40-49",
    "age_band_min_age": 40,
    "age_band_max_age": 49,
    "number_of_products": 2,
    "credit_score": 650.0,
    "age": 42.0,
    "tenure": 7.0,
    "balance": 0.0,
    "has_credit_card": 1,
    "is_active_member": 0,
    "estimated_salary": 184843.77,
}


@dataclass
class FakeResult:
    probability: float = 0.81
    decision: str = "contactar"
    action: str = "revision_humana_retencion"
    model_version: str | None = "1"
    run_id: str | None = "registered-run"
    used_fallback: bool = False
    fallback_reason: str | None = None


class FakeProvider:
    model_name = "pset3-churn-classifier"
    alias = "champion"

    def model_info(self) -> dict[str, Any]:
        return {
            "service_status": "ok",
            "model_loaded": True,
            "model_name": self.model_name,
            "alias": self.alias,
            "version": "1",
            "run_id": "registered-run",
            "error": None,
        }

    def refresh(self, force: bool = False) -> None:
        return None

    def infer(self, features: Any) -> FakeResult:
        return FakeResult()


class FakeStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def write_many(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(records)

def setup_function() -> None:
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_prediction_store] = lambda: FakeStore()
    app.state.request_metrics = []


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_valid_input_returns_contract_and_logs_prediction() -> None:
    store = FakeStore()
    app.dependency_overrides[get_prediction_store] = lambda: store

    with TestClient(app) as client:
        response = client.post(
            "/predecir",
            json={"features": VALID_FEATURES, "observation_id": "obs-42"},
            headers={"X-Batch-ID": "week-01"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["probabilidad"] == 0.81
    assert body["decision"] == "contactar"
    assert body["model_version"] == "1"
    assert body["run_id"] == "registered-run"
    assert body["observation_id"] == "obs-42"
    assert body["used_fallback"] is False
    assert len(store.records) == 1
    assert store.records[0]["batch_id"] == "week-01"


def test_missing_field_returns_422_with_field_location() -> None:
    features = VALID_FEATURES.copy()
    del features["age"]
    with TestClient(app) as client:
        response = client.post("/predecir", json={"features": features})

    assert response.status_code == 422
    assert "age" in str(response.json()["detail"])


def test_wrong_type_returns_422() -> None:
    features = VALID_FEATURES | {"credit_score": "high"}
    with TestClient(app) as client:
        response = client.post("/predecir", json={"features": features})

    assert response.status_code == 422
    assert "credit_score" in str(response.json()["detail"])


def test_unknown_category_returns_422() -> None:
    features = VALID_FEATURES | {"geography": "Atlantis"}
    with TestClient(app) as client:
        response = client.post("/predecir", json={"features": features})

    assert response.status_code == 422
    assert "geography" in str(response.json()["detail"])


def test_inconsistent_age_band_returns_422() -> None:
    features = VALID_FEATURES | {"age_band": "18-29"}
    with TestClient(app) as client:
        response = client.post("/predecir", json={"features": features})

    assert response.status_code == 422
    assert "age_band" in str(response.json()["detail"])


def test_system_metrics_expose_latency_and_validation_error_rate() -> None:
    with TestClient(app) as client:
        invalid = client.post("/predecir", json={"features": {"geography": "Atlantis"}})
        metrics = client.get("/metricas/sistema")

    assert invalid.status_code == 422
    assert metrics.status_code == 200
    assert metrics.json()["request_count"] >= 1
    assert metrics.json()["error_4xx_rate"] > 0
    assert metrics.json()["p50_latency_ms"] >= 0


def test_missing_registry_model_uses_explicit_dummy_fallback(
    monkeypatch,
) -> None:
    class UnavailableRegistry:
        def get_model_version_by_alias(self, model_name: str, alias: str) -> None:
            raise ConnectionError("MLflow is unavailable.")

    provider = RegistryModelProvider(tracking_uri="http://mlflow.invalid")
    provider.client = UnavailableRegistry()
    app.dependency_overrides[get_model_provider] = lambda: provider

    with TestClient(app) as client:
        response = client.post(
            "/predecir",
            json={"features": VALID_FEATURES, "observation_id": "fallback-case"},
        )

    assert response.status_code == 200
    assert response.json()["used_fallback"] is True
    assert response.json()["fallback_reason"]
    assert response.json()["probabilidad"] == DUMMY_PRIOR_PROBABILITY


def test_manual_review_mode_routes_every_prediction_to_human_review(
    monkeypatch,
) -> None:
    registry_calls: list[str] = []

    class UnexpectedRegistry:
        def get_model_version_by_alias(self, model_name: str, alias: str) -> None:
            registry_calls.append(f"{model_name}@{alias}")
            raise AssertionError("manual-review mode must not load a model")

    provider = RegistryModelProvider(tracking_uri="http://mlflow.invalid")
    provider.client = UnexpectedRegistry()
    store = FakeStore()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_prediction_store] = lambda: store
    monkeypatch.setenv("PREDICTION_MODE", "manual_review_all")

    with TestClient(app) as client:
        model_info = client.get("/modelo").json()
        response = client.post(
            "/predecir",
            json={"features": VALID_FEATURES, "observation_id": "manual-review"},
        )

    assert response.status_code == 200
    assert response.json()["decision"] == "revision_humana"
    assert response.json()["accion"] == "revision_humana_retencion"
    assert response.json()["used_fallback"] is True
    assert response.json()["model_version"] is None
    assert response.json()["run_id"] is None
    assert response.json()["fallback_reason"] == (
        "PREDICTION_MODE=manual_review_all"
    )
    assert model_info["service_status"] == "degraded"
    assert model_info["model_loaded"] is False
    assert registry_calls == []
    assert store.records[0]["decision"] == "revision_humana"


def test_batch_endpoint_writes_all_predictions_with_shared_batch_id() -> None:
    store = FakeStore()
    app.dependency_overrides[get_prediction_store] = lambda: store
    with TestClient(app) as client:
        response = client.post(
            "/predecir/lote",
            json={
                "batch_id": "week-01",
                "obt_snapshot_id": "snapshot-001",
                "predictions": [
                    {"features": VALID_FEATURES, "observation_id": "obs-1"},
                    {"features": VALID_FEATURES, "observation_id": "obs-2"},
                ],
            },
        )

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert len(store.records) == 2
    assert {record["batch_id"] for record in store.records} == {"week-01"}
    assert {record["obt_snapshot_id"] for record in store.records} == {"snapshot-001"}


def test_batch_endpoint_reports_snowflake_persistence_failure() -> None:
    class UnavailableStore(FakeStore):
        def write_many(self, records: list[dict[str, Any]]) -> None:
            raise ConnectionError("Snowflake unavailable")

    app.dependency_overrides[get_prediction_store] = UnavailableStore
    with TestClient(app) as client:
        response = client.post(
            "/predecir/lote",
            json={
                "batch_id": "week-01",
                "predictions": [{"features": VALID_FEATURES}],
            },
        )

    assert response.status_code == 503
    assert "no se pudo registrar en Snowflake" in response.json()["detail"]


def test_model_endpoint_reflects_alias_move_without_replacing_provider(
    monkeypatch,
) -> None:
    class SwitchableRegistry:
        version = "1"

        def get_model_version_by_alias(self, model_name: str, alias: str) -> Any:
            return SimpleNamespace(version=self.version, run_id=f"run-{self.version}")

    class FakeModel:
        classes_ = [0, 1]

        def predict_proba(self, frame: Any) -> np.ndarray:
            return np.tile([0.8, 0.2], (len(frame), 1))

    provider = RegistryModelProvider(tracking_uri="http://mlflow.invalid")
    registry = SwitchableRegistry()
    provider.client = registry
    monkeypatch.setattr(api_module.mlflow.sklearn, "load_model", lambda uri: FakeModel())
    app.dependency_overrides[get_model_provider] = lambda: provider

    with TestClient(app) as client:
        first = client.get("/modelo").json()
        registry.version = "2"
        second = client.get("/modelo").json()

    assert first["version"] == "1"
    assert first["run_id"] == "run-1"
    assert second["version"] == "2"
    assert second["run_id"] == "run-2"


def test_failed_alias_refresh_reports_degraded_while_serving_last_good_model(
    monkeypatch,
) -> None:
    class SwitchableRegistry:
        version = "1"

        def get_model_version_by_alias(self, model_name: str, alias: str) -> Any:
            return SimpleNamespace(version=self.version, run_id=f"run-{self.version}")

    class FakeModel:
        classes_ = [0, 1]

        def predict_proba(self, frame: Any) -> np.ndarray:
            return np.tile([0.8, 0.2], (len(frame), 1))

    def load_model(uri: str) -> FakeModel:
        if uri.endswith("/2"):
            raise RuntimeError("model version 2 is unavailable")
        return FakeModel()

    provider = RegistryModelProvider(tracking_uri="http://mlflow.invalid")
    registry = SwitchableRegistry()
    provider.client = registry
    monkeypatch.setattr(api_module.mlflow.sklearn, "load_model", load_model)
    app.dependency_overrides[get_model_provider] = lambda: provider

    with TestClient(app) as client:
        first = client.get("/modelo").json()
        registry.version = "2"
        second = client.get("/modelo").json()
        prediction = client.post(
            "/predecir",
            json={"features": VALID_FEATURES},
        ).json()

    assert first["service_status"] == "ok"
    assert second["service_status"] == "degraded"
    assert second["version"] == "1"
    assert "model version 2 is unavailable" in second["error"]
    assert prediction["model_version"] == "1"
    assert prediction["used_fallback"] is False
    assert prediction["fallback_reason"]
