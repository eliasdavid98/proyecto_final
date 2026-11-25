from __future__ import annotations

import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _sample() -> dict[str, object]:
    return json.loads((FIXTURES / "champion_sample.json").read_text())


def _model():
    return mlflow.sklearn.load_model(str(FIXTURES / "champion_model"))


def test_exported_champion_reproduces_fixed_reference_predictions() -> None:
    sample = _sample()
    model = _model()
    features = pd.DataFrame(sample["features"])

    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    labels = model.predict(features)

    assert probabilities.tolist() == pytest.approx(
        sample["expected_probabilities"], rel=0, abs=1e-12
    )
    assert labels.tolist() == sample["expected_labels"]
    assert sample["version"] == "1"
    assert sample["run_id"] == "fa804ba98b5b4b02acf59f02b82e81cd"


def test_exported_challenger_reproduces_fixed_reference_predictions() -> None:
    sample = _sample()
    model = mlflow.sklearn.load_model(
        str(FIXTURES / "challenger_model")
    )
    features = pd.DataFrame(sample["features"])

    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    labels = model.predict(features)

    assert probabilities.tolist() == pytest.approx(
        sample["challenger_probabilities"], rel=0, abs=1e-12
    )
    assert labels.tolist() == sample["challenger_labels"]
    assert sample["challenger_version"] == "2"
    assert sample["challenger_run_id"] == "97afc921649a44199dada5ede3bc78a1"


def test_higher_age_increases_risk_for_the_fixed_comparison_pair() -> None:
    model = _model()
    sample = _sample()
    young = sample["features"][0] | {
        "age": 30.0,
        "age_band": "30-39",
        "age_band_min_age": 30,
        "age_band_max_age": 39,
    }
    older = sample["features"][0] | {
        "age": 60.0,
        "age_band": "50+",
        "age_band_min_age": 50,
        "age_band_max_age": 100,
    }
    probabilities = model.predict_proba(pd.DataFrame([young, older]))[:, 1]

    assert probabilities[1] > probabilities[0]


def test_active_member_has_lower_risk_for_the_fixed_comparison_pair() -> None:
    model = _model()
    sample = _sample()
    inactive = sample["features"][0] | {"is_active_member": 0.0}
    active = sample["features"][0] | {"is_active_member": 1.0}
    probabilities = model.predict_proba(pd.DataFrame([inactive, active]))[:, 1]

    assert probabilities[1] < probabilities[0]
