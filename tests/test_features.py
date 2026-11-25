import unittest
from decimal import Decimal

import pandas as pd
from sklearn.base import clone

from src.features import ChurnFeatureEngineer


class ChurnFeatureEngineerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                "age": [18, 39, 40, 100, 101, None],
                "balance": [0, 0.01, 100, -1, None, 0],
                "age_band": ["18-29", "30-39", "40-49", "50+", "50+", None],
                "age_band_min_age": [18, 30, 40, 50, 50, None],
                "age_band_max_age": [29, 39, 49, 100, 100, None],
                "credit_score": [600, 650, 700, 750, 800, 620],
            }
        )

    def test_derives_inference_time_categories_and_drops_duplicate_age_band(self) -> None:
        transformed = ChurnFeatureEngineer().fit_transform(self.features)

        self.assertEqual(
            transformed["age_40_plus"].tolist(),
            [
                "menor_de_40",
                "menor_de_40",
                "40_o_mas",
                "40_o_mas",
                "fuera_de_rango",
                "faltante",
            ],
        )
        self.assertEqual(
            transformed["balance_status"].tolist(),
            [
                "saldo_cero",
                "saldo_positivo",
                "saldo_positivo",
                "negativo_invalido",
                "faltante",
                "saldo_cero",
            ],
        )
        self.assertNotIn("age_band", transformed.columns)
        self.assertNotIn("age_band_min_age", transformed.columns)
        self.assertNotIn("age_band_max_age", transformed.columns)
        self.assertIn("credit_score", transformed.columns)

    def test_fit_does_not_depend_on_target_values(self) -> None:
        transformer_a = clone(ChurnFeatureEngineer()).fit(
            self.features, pd.Series([0, 0, 0, 1, 1, 1])
        )
        transformer_b = clone(ChurnFeatureEngineer()).fit(
            self.features, pd.Series([1, 1, 1, 0, 0, 0])
        )

        pd.testing.assert_frame_equal(
            transformer_a.transform(self.features),
            transformer_b.transform(self.features),
        )

    def test_missing_required_feature_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "balance"):
            ChurnFeatureEngineer().fit(self.features.drop(columns="balance"))

    def test_snowflake_decimal_columns_are_numeric_after_transform(self) -> None:
        features = pd.DataFrame(
            {
                "age": [Decimal("40"), Decimal("39")],
                "balance": [Decimal("0"), Decimal("100.25")],
                "credit_score": [Decimal("700"), Decimal("650")],
            }
        )

        transformed = ChurnFeatureEngineer().fit_transform(features)

        for column in ("age", "balance", "credit_score"):
            self.assertTrue(pd.api.types.is_numeric_dtype(transformed[column]))


if __name__ == "__main__":
    unittest.main()
