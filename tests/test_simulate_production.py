import pandas as pd

from scripts.simulate_production import _canonicalize


def test_canonicalize_accepts_the_dataset_lowercase_id_header() -> None:
    source = pd.DataFrame(
        {
            "id": [123],
            "CreditScore": [650],
            "Geography": ["France"],
            "Gender": ["Female"],
            "Age": [42],
            "Tenure": [7],
            "Balance": [0.0],
            "NumOfProducts": [2],
            "HasCrCard": [1],
            "IsActiveMember": [0],
            "EstimatedSalary": [100000.0],
        }
    )

    result = _canonicalize(source)

    assert result.loc[0, "observation_id"] == "123"
    assert result.loc[0, "age_band"] == "40-49"
    assert pd.isna(result.loc[0, "exited"])
