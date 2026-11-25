import unittest
from decimal import Decimal

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.preparacion import make_model_pipeline, make_stratified_cv, split_churn_data


def sample_obt() -> pd.DataFrame:
    train = pd.DataFrame(
        {
            "observation_id": [f"train-{index}" for index in range(60)],
            "source_split": "train",
            "customer_id": range(60),
            "customer_id_raw": [str(index) for index in range(60)],
            "surname": [f"name-{index}" for index in range(60)],
            "geography": ["France", "Spain", "Germany"] * 20,
            "gender": ["Female", "Male"] * 30,
            "age_band": ["18-29"] * 60,
            "age_band_min_age": [18] * 60,
            "age_band_max_age": [29] * 60,
            "credit_score": [Decimal(600 + index) for index in range(60)],
            "age": [Decimal(20 + index % 50) for index in range(60)],
            "tenure": [Decimal(index % 10) for index in range(60)],
            "balance": [Decimal(index * 10) for index in range(60)],
            "number_of_products": [Decimal(1 + index % 4) for index in range(60)],
            "has_credit_card": [Decimal(index % 2) for index in range(60)],
            "is_active_member": [Decimal((index + 1) % 2) for index in range(60)],
            "estimated_salary": [Decimal(1000 + index * 100) for index in range(60)],
            "exited": [index % 2 for index in range(60)],
        }
    )
    test = train.iloc[:12].copy()
    test["observation_id"] = [f"test-{index}" for index in range(12)]
    test["source_split"] = "test"
    test["exited"] = None
    return pd.concat([train, test], ignore_index=True)


class ChurnPreparationTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_excludes_competition_test(self) -> None:
        obt = sample_obt()
        split = split_churn_data(obt)
        repeated = split_churn_data(obt)

        self.assertEqual(len(split.X_development), 48)
        self.assertEqual(len(split.X_holdout), 12)
        self.assertEqual(len(split.X_competition_test), 12)
        self.assertEqual(split.y_development.mean(), 0.5)
        self.assertEqual(split.y_holdout.mean(), 0.5)
        self.assertNotIn("exited", split.X_development.columns)
        self.assertNotIn("exited", split.X_competition_test.columns)
        pd.testing.assert_frame_equal(split.X_development, repeated.X_development)
        pd.testing.assert_series_equal(split.y_holdout, repeated.y_holdout)

    def test_full_pipeline_fits_snowflake_decimal_values_and_ignores_unseen_category(
        self,
    ) -> None:
        split = split_churn_data(sample_obt())
        pipeline = make_model_pipeline(DummyClassifier(strategy="prior"))
        pipeline.fit(split.X_development, split.y_development)

        unseen_category = split.X_holdout.copy()
        unseen_category["geography"] = "Never-Seen"
        predictions = pipeline.predict(unseen_category)

        self.assertEqual(len(predictions), len(split.y_holdout))
        numeric_columns = pipeline.named_steps["preprocessing"].transformers_[0][2]
        self.assertIn("credit_score", numeric_columns)

    def test_stratified_cv_has_disjoint_folds(self) -> None:
        split = split_churn_data(sample_obt())
        cv = make_stratified_cv()
        folds = list(cv.split(split.X_development, split.y_development))

        self.assertEqual(len(folds), 3)
        for train_indices, validation_indices in folds:
            self.assertFalse(set(train_indices).intersection(validation_indices))
            self.assertEqual(
                split.y_development.iloc[validation_indices].nunique(), 2
            )

    def test_labeled_competition_test_is_rejected(self) -> None:
        obt = sample_obt()
        obt.loc[obt["source_split"].eq("test").idxmax(), "exited"] = 1

        with self.assertRaisesRegex(ValueError, "sin etiqueta"):
            split_churn_data(obt)


if __name__ == "__main__":
    unittest.main()
