from __future__ import annotations

from typing import Any

import api.main as api_module
from api.main import SnowflakePredictionStore


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...] | None]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(
        self, statement: str, parameters: tuple[Any, ...] | None = None
    ) -> None:
        self.statements.append((statement, parameters))


class FakeConnection:
    def __init__(self) -> None:
        self.fake_cursor = FakeCursor()
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def close(self) -> None:
        self.closed = True


def _record(index: int) -> dict[str, Any]:
    return {
        "prediction_id": f"prediction-{index}",
        "observation_id": f"observation-{index}",
        "inputs": {"balance": 0.0},
        "probability": 0.2,
        "decision": "sin_contacto",
        "action": "sin_accion",
        "model_name": "churn",
        "model_alias": "champion",
        "model_version": "1",
        "run_id": "run-1",
        "fallback_reason": None,
        "latency_ms": 1.0,
        "used_fallback": False,
        "batch_id": "batch-1",
        "obt_snapshot_id": "snapshot-1",
        "source_gold_tables": "FACT_CUSTOMER_CHURN",
    }


def test_write_many_uses_bounded_multirow_inserts(monkeypatch) -> None:
    for key, value in {
        "SNOWFLAKE_ACCOUNT": "account",
        "SNOWFLAKE_USERNAME": "user",
        "SNOWFLAKE_PASSWORD": "password",
        "SNOWFLAKE_DATABASE": "CHURN_DB",
        "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
    }.items():
        monkeypatch.setenv(key, value)
    connection = FakeConnection()
    monkeypatch.setattr(
        api_module.snowflake.connector, "connect", lambda **kwargs: connection
    )

    SnowflakePredictionStore().write_many([_record(index) for index in range(501)])

    inserts = [
        (statement, parameters)
        for statement, parameters in connection.fake_cursor.statements
        if "INSERT INTO" in statement
    ]
    assert connection.closed
    assert len(inserts) == 2
    assert "FROM VALUES" in inserts[0][0]
    assert len(inserts[0][1] or ()) == 500 * 16
    assert len(inserts[1][1] or ()) == 16
    statements = [statement for statement, _ in connection.fake_cursor.statements]
    assert statements.index("BEGIN") < statements.index(inserts[0][0])
    assert statements.index("COMMIT") > statements.index(inserts[1][0])
