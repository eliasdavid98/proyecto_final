import pandas as pd

from monitoreo.alertas import build_drift_alerts, performance_below_threshold
from monitoreo.deriva import compare_feature_distributions


def test_psi_and_statistical_test_flag_an_induced_balance_shift() -> None:
    reference = pd.DataFrame({"balance": [0.0] * 80 + [5000.0] * 20})
    altered = pd.DataFrame({"balance": [50000.0] * 100})

    metrics = compare_feature_distributions(
        reference, altered, numeric_features=["balance"], categorical_features=[]
    )
    alerts = build_drift_alerts("week-08", metrics)

    assert metrics[0]["psi"] > 0.25
    assert metrics[0]["statistical_test"] == "kolmogorov_smirnov"
    assert alerts[0]["severity"] == "alta"


def test_unaltered_distribution_does_not_trigger_psi_alert() -> None:
    reference = pd.DataFrame({"geography": ["France", "Germany", "Spain"] * 20})
    current = reference.iloc[:30].reset_index(drop=True)

    metrics = compare_feature_distributions(
        reference,
        current,
        numeric_features=[],
        categorical_features=["geography"],
    )

    assert build_drift_alerts("week-01", metrics) == []


def test_performance_trigger_requires_a_real_metric() -> None:
    assert performance_below_threshold(None, cv_mean=0.82, cv_std=0.01) is False
    assert performance_below_threshold(0.79, cv_mean=0.82, cv_std=0.01) is True
