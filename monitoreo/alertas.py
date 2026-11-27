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


def build_concept_drift_alerts(
    batch_id: str,
    roc_auc: float | None,
    average_precision: float | None,
    cv_roc_auc_mean: float = 0.9317,
    cv_roc_auc_std: float = 0.0026,
    cv_ap_mean: float = 0.8198,
    cv_ap_std: float = 0.0039,
    thresholds: AlertThresholds = AlertThresholds(),
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    checks = [
        ("ROC_AUC", roc_auc, cv_roc_auc_mean, cv_roc_auc_std),
        ("AVERAGE_PRECISION", average_precision, cv_ap_mean, cv_ap_std),
    ]
    for metric_name, value, mean, std in checks:
        if value is None:
            continue
        limit = mean - thresholds.concept_drift_sigma * std
        if value < limit:
            alerts.append(
                {
                    "alert_id": f"{batch_id}:{metric_name.lower()}:concept_drift",
                    "batch_id": batch_id,
                    "metric": metric_name,
                    "value": float(value),
                    "threshold": float(limit),
                    "window": "1 lote semanal sintético (contra CV PSet 3)",
                    "severity": "alta",
                    "recipient": "Equipo de datos y negocio",
                    "action": (
                        "Evaluar deriva conceptual y verificar si se cumplen "
                        f"{thresholds.consecutive_performance_batches} lotes "
                        "consecutivos para disparar reentrenamiento."
                    ),
                    "supporting_test": "cv_mean_minus_2sd",
                    "supporting_p_value": None,
                }
            )
    return alerts