"""Reglas transparentes de alerta del PSet 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlertThresholds:
    psi_warning: float = 0.10
    psi_high: float = 0.25
    concept_drift_sigma: float = 2.0
    consecutive_performance_batches: int = 2


def build_drift_alerts(
    batch_id: str,
    feature_metrics: list[dict[str, Any]],
    thresholds: AlertThresholds = AlertThresholds(),
) -> list[dict[str, Any]]:
    alerts = []
    for metric in feature_metrics:
        psi = float(metric["psi"])
        if psi <= thresholds.psi_warning:
            continue
        high = psi > thresholds.psi_high
        alerts.append(
            {
                "alert_id": f"{batch_id}:{metric['feature']}:psi",
                "batch_id": batch_id,
                "metric": f"PSI({metric['feature']})",
                "value": psi,
                "threshold": (
                    thresholds.psi_high if high else thresholds.psi_warning
                ),
                "window": "1 lote semanal sintético",
                "severity": "alta" if high else "media",
                "recipient": "Equipo de datos",
                "action": (
                    "Revisar la alteración inducida y validar la fuente; "
                    "no reentrenar sin etiquetas disponibles."
                    if high
                    else "Inspeccionar la distribución contra la referencia."
                ),
                "supporting_test": metric.get("statistical_test"),
                "supporting_p_value": metric.get("p_value"),
            }
        )
    return alerts


def performance_below_threshold(
    metric_value: float | None,
    cv_mean: float,
    cv_std: float,
    sigma: float = 2.0,
) -> bool:
    if metric_value is None:
        return False
    return metric_value < cv_mean - sigma * cv_std
