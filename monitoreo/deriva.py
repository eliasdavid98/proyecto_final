"""PSI, KS y chi-cuadrado para comparar lotes con train."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

PSI_EPSILON = 1e-4
NUMERIC_RANGES = {
    "credit_score": (300.0, 850.0),
    "age": (18.0, 100.0),
    "tenure": (0.0, 10.0),
    "balance": (0.0, None),
    "number_of_products": (1.0, 4.0),
    "has_credit_card": (0.0, 1.0),
    "is_active_member": (0.0, 1.0),
    "estimated_salary": (0.0, None),
}


def _psi(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    expected_share = expected_counts / max(expected_counts.sum(), 1)
    actual_share = actual_counts / max(actual_counts.sum(), 1)
    expected_share = np.clip(expected_share, PSI_EPSILON, None)
    actual_share = np.clip(actual_share, PSI_EPSILON, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def _numeric_bins(expected: pd.Series) -> np.ndarray:
    values = pd.to_numeric(expected, errors="coerce").dropna().to_numpy(dtype=float)
    if not values.size:
        return np.array([-np.inf, np.inf])
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    inner = quantiles[1:-1]
    return np.concatenate(([-np.inf], inner, [np.inf]))


def compare_numeric(
    feature: str,
    expected: pd.Series,
    actual: pd.Series,
) -> dict[str, Any]:
    reference = pd.to_numeric(expected, errors="coerce").dropna()
    current = pd.to_numeric(actual, errors="coerce").dropna()
    edges = _numeric_bins(reference)
    expected_counts = np.histogram(reference, bins=edges)[0].astype(float)
    actual_counts = np.histogram(current, bins=edges)[0].astype(float)
    if reference.empty or current.empty:
        statistic = p_value = float("nan")
    else:
        test = ks_2samp(reference, current, alternative="two-sided", method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    lower, upper = NUMERIC_RANGES.get(feature, (None, None))
    out_of_range = pd.Series(False, index=actual.index)
    numeric_actual = pd.to_numeric(actual, errors="coerce")
    if lower is not None:
        out_of_range |= numeric_actual.lt(lower)
    if upper is not None:
        out_of_range |= numeric_actual.gt(upper)
    return {
        "feature": feature,
        "feature_type": "numerica",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "psi": _psi(expected_counts, actual_counts),
        "statistical_test": "kolmogorov_smirnov",
        "test_statistic": statistic,
        "p_value": p_value,
        "missing_rate": float(actual.isna().mean()),
        "out_of_range_rate": float(out_of_range.mean()),
        "unseen_category_rate": 0.0,
    }


def compare_categorical(
    feature: str,
    expected: pd.Series,
    actual: pd.Series,
) -> dict[str, Any]:
    reference = expected.astype("string").fillna("<MISSING>")
    current = actual.astype("string").fillna("<MISSING>")
    categories = sorted(set(reference).union(set(current)))
    expected_counts = reference.value_counts().reindex(categories, fill_value=0)
    actual_counts = current.value_counts().reindex(categories, fill_value=0)
    contingency = np.vstack((expected_counts.to_numpy(), actual_counts.to_numpy()))
    if contingency.shape[1] > 1:
        statistic, p_value, _, _ = chi2_contingency(contingency, correction=False)
        statistic, p_value = float(statistic), float(p_value)
    else:
        statistic, p_value = 0.0, 1.0
    known = set(reference.dropna().unique())
    unseen_rate = float((~current.isin(known | {"<MISSING>"})).mean())
    return {
        "feature": feature,
        "feature_type": "categorica",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "psi": _psi(
            expected_counts.to_numpy(dtype=float),
            actual_counts.to_numpy(dtype=float),
        ),
        "statistical_test": "chi_square",
        "test_statistic": statistic,
        "p_value": p_value,
        "missing_rate": float(actual.isna().mean()),
        "out_of_range_rate": 0.0,
        "unseen_category_rate": unseen_rate,
    }


def compare_feature_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> list[dict[str, Any]]:
    missing = (set(numeric_features) | set(categorical_features)) - set(reference.columns)
    missing |= (set(numeric_features) | set(categorical_features)) - set(current.columns)
    if missing:
        raise ValueError(f"Faltan features para monitoreo: {sorted(missing)}")
    results = [
        compare_numeric(feature, reference[feature], current[feature])
        for feature in numeric_features
    ]
    results.extend(
        compare_categorical(feature, reference[feature], current[feature])
        for feature in categorical_features
    )
    return results
