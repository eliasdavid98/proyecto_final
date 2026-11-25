"""Simula ocho lotes semanales sintéticos sobre el test sin etiquetas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
AGE_BANDS = ((18, 29, "18-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 100, "50+"))
DEFAULT_START = date(2026, 1, 5)


def _read_competition_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta data.py para obtener los CSV locales."
        )
    return pd.read_csv(path)


def _canonicalize(source: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "id": "observation_id",
        "CreditScore": "credit_score",
        "Geography": "geography",
        "Gender": "gender",
        "Age": "age",
        "Tenure": "tenure",
        "Balance": "balance",
        "NumOfProducts": "number_of_products",
        "HasCrCard": "has_credit_card",
        "IsActiveMember": "is_active_member",
        "EstimatedSalary": "estimated_salary",
        "Exited": "exited",
    }
    frame = source.rename(columns=rename).copy()
    if "observation_id" not in frame and "Id" in source:
        frame = source.rename(columns={"Id": "observation_id", **rename}).copy()
    required = set(rename.values()) - {"exited"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV de entrada sin columnas requeridas: {sorted(missing)}")
    frame["observation_id"] = frame["observation_id"].astype(str)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    for lower, upper, label in AGE_BANDS:
        mask = frame["age"].between(lower, upper, inclusive="both")
        frame.loc[mask, "age_band"] = label
        frame.loc[mask, "age_band_min_age"] = lower
        frame.loc[mask, "age_band_max_age"] = upper
    frame["balance_status"] = np.select(
        [
            frame["balance"].isna(),
            pd.to_numeric(frame["balance"], errors="coerce").lt(0),
            pd.to_numeric(frame["balance"], errors="coerce").eq(0),
        ],
        ["faltante", "negativo_invalido", "saldo_cero"],
        default="saldo_positivo",
    )
    if frame["age_band"].isna().any():
        raise ValueError("Hay edades sin banda válida en los datos de simulación.")
    if "exited" not in frame:
        frame["exited"] = pd.NA
    frame["source_split"] = "test"
    return frame


def _load_model_fixture() -> tuple[Any, dict[str, Any]]:
    metadata_path = ROOT / "tests" / "fixtures" / "champion_sample.json"
    metadata = json.loads(metadata_path.read_text())
    model_path = ROOT / "tests" / "fixtures" / "champion_model"
    return mlflow.sklearn.load_model(str(model_path)), metadata


def _induce_drift(batch: pd.DataFrame) -> pd.DataFrame:
    changed = batch.copy()
    changed["balance"] = pd.to_numeric(changed["balance"], errors="coerce") + 50000.0
    changed["geography"] = "Germany"
    changed["balance_status"] = "saldo_positivo"
    return changed


def simulate(
    source_path: Path,
    batch_count: int = 8,
    output_dir: Path = ROOT / "docs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if batch_count < 8:
        raise ValueError("PSet 4 requiere al menos ocho lotes.")

    reference = _canonicalize(_read_competition_csv(ROOT / "data" / "train.csv"))
    source = _canonicalize(_read_competition_csv(source_path))
    model, model_metadata = _load_model_fixture()
    splits = np.array_split(np.arange(len(source)), batch_count)
    prediction_rows: list[dict[str, Any]] = []
    monitoring_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    thresholds = AlertThresholds()

    for index, row_indices in enumerate(splits):
        batch_id = f"synthetic-week-{index + 1:02d}"
        batch = source.iloc[row_indices].copy()
        induced = index == batch_count - 1
        if induced:
            batch = _induce_drift(batch)

        model_frame = batch.loc[:, MODEL_INPUTS].copy()
        for column in NUMERIC_FEATURES:
            model_frame[column] = pd.to_numeric(
                model_frame[column], errors="coerce"
            ).astype("float64")

        started = time.perf_counter()
        probabilities = model.predict_proba(model_frame)[:, list(model.classes_).index(1)]
        elapsed_ms = (time.perf_counter() - started) * 1000
        per_row_ms = elapsed_ms / max(len(batch), 1)
        decisions = probabilities >= 0.5
        period_start = DEFAULT_START + timedelta(days=index * 7)
        for position, (_, row) in enumerate(batch.iterrows()):
            prediction_rows.append(
                {
                    "prediction_id": f"{batch_id}-{position + 1:05d}",
                    "created_at_synthetic": period_start.isoformat(),
                    "batch_id": batch_id,
                    "probability": float(probabilities[position]),
                    "decision": "contactar" if decisions[position] else "sin_contacto",
                    "action": (
                        "revision_humana_retencion"
                        if decisions[position]
                        else "sin_accion"
                    ),
                    "model_name": model_metadata["model_name"],
                    "model_alias": "champion",
                    "model_version": model_metadata["version"],
                    "run_id": model_metadata["run_id"],
                    "latency_ms_offline": round(per_row_ms, 6),
                    "used_fallback": False,
                    "obt_snapshot_id": "competition-test-row-order-not-a-snapshot",
                    "source_gold_tables": (
                        "FACT_CUSTOMER_CHURN,DIM_GEOGRAPHY,DIM_GENDER,"
                        "DIM_AGE_BAND,DIM_PRODUCT_COUNT"
                    ),
                    "target": None,
                    "outcome_available": False,
                    "simulation_basis": "fila_de_archivo_sin_orden_temporal",
                    "induced_drift": induced,
                }
            )

        feature_metrics = compare_feature_distributions(
            reference,
            batch,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        alerts = build_drift_alerts(batch_id, feature_metrics, thresholds)
        alert_rows.extend(alerts)
        monitoring_rows.append(
            {
                "batch_id": batch_id,
                "synthetic_period_start": period_start.isoformat(),
                "rows": int(len(batch)),
                "simulation_basis": "fila_de_archivo_sin_orden_temporal",
                "induced_drift": induced,
                "induced_change": (
                    "balance += 50000; geography='Germany'"
                    if induced
                    else None
                ),
                "mean_probability": float(np.mean(probabilities)),
                "predicted_contact_rate": float(np.mean(decisions)),
                "target_available": False,
                "roc_auc": None,
                "average_precision": None,
                "f2_at_0_5": None,
                "api_request_count": None,
                "api_p50_latency_ms": None,
                "api_p95_latency_ms": None,
                "api_4xx_rate": None,
                "api_5xx_rate": None,
                "offline_model_scoring_p50_latency_ms": per_row_ms,
                "max_psi": max(metric["psi"] for metric in feature_metrics),
                "feature_metrics_json": json.dumps(
                    feature_metrics, ensure_ascii=False
                ),
                "alert_count": len(alerts),
                "alert_ids": ",".join(alert["alert_id"] for alert in alerts),
                "retraining_triggered": False,
                "retraining_block_reason": (
                    "target no disponible: test de competencia sin etiquetas"
                ),
            }
        )

    retraining = pd.DataFrame(
        [
            {
                "retraining_id": "synthetic-week-08-no-labels",
                "trigger_batch_id": f"synthetic-week-{batch_count:02d}",
                "trigger": "PSI > 0.25",
                "status": "blocked",
                "target_available": False,
                "candidate_run_id": None,
                "evaluation_metric": None,
                "promotion": "not_evaluated",
                "decision": "no_retraining",
                "reason": (
                    "La fuente no tiene etiqueta para el lote y no hay observaciones "
                    "posteriores etiquetadas. Se bloquea el reentrenamiento para no "
                    "inventar targets ni comparar con el holdout ya consumido."
                ),
            }
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame = pd.DataFrame(prediction_rows)
    monitoring_frame = pd.DataFrame(monitoring_rows)
    alerts_frame = pd.DataFrame(alert_rows)
    predictions_frame.to_csv(output_dir / "predicciones.csv", index=False)
    monitoring_frame.to_csv(output_dir / "monitoreo.csv", index=False)
    alerts_frame.to_csv(output_dir / "alertas.csv", index=False)
    retraining.to_csv(output_dir / "reentrenamientos.csv", index=False)
    return predictions_frame, monitoring_frame, alerts_frame, retraining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "test.csv",
        help="CSV local de competencia sin etiqueta (por defecto data/test.csv).",
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    predictions, monitoring, alerts, retraining = simulate(
        args.source, batch_count=args.batches, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "prediction_rows": len(predictions),
                "batch_count": len(monitoring),
                "alerts": len(alerts),
                "retraining_status": retraining.iloc[0]["status"],
                "induced_drift_batch": monitoring.iloc[-1]["batch_id"],
                "output_dir": str(args.output_dir),
                "warning": (
                    "The batches are row-order slices, not temporal cohorts; "
                    "targets are unavailable."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
