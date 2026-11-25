"""Features derivadas disponibles antes de conocer la etiqueta de churn."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deriva segmentos de edad y balance sin aprender de la etiqueta."""

    required_columns = ("age", "balance")
    redundant_age_columns = ("age_band", "age_band_min_age", "age_band_max_age")
    numeric_columns = (
        "credit_score",
        "age",
        "tenure",
        "balance",
        "number_of_products",
        "has_credit_card",
        "is_active_member",
        "estimated_salary",
    )

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Valida las entradas y registra sus nombres; no calcula estadísticas."""
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(
            [
                column
                for column in X.columns
                if column not in self.redundant_age_columns
            ]
            + ["age_40_plus", "balance_status"],
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Añade variables derivadas y elimina la banda de edad duplicada."""
        check_is_fitted(self, attributes=["feature_names_in_", "feature_names_out_"])
        self._validate_input(X)

        result = X.drop(columns=list(self.redundant_age_columns), errors="ignore").copy()
        for column in self.numeric_columns:
            if column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce")

        age = result["age"]
        age_status = np.select(
            [
                age.isna(),
                age.lt(18) | age.gt(100),
                age.ge(40),
            ],
            ["faltante", "fuera_de_rango", "40_o_mas"],
            default="menor_de_40",
        )
        result["age_40_plus"] = pd.Series(age_status, index=X.index, dtype="string")

        balance = result["balance"]
        balance_status = np.select(
            [
                balance.isna(),
                balance.lt(0),
                balance.eq(0),
            ],
            ["faltante", "negativo_invalido", "saldo_cero"],
            default="saldo_positivo",
        )
        result["balance_status"] = pd.Series(
            balance_status, index=X.index, dtype="string"
        )
        return result

    def get_feature_names_out(
        self, input_features: list[str] | np.ndarray | None = None
    ) -> np.ndarray:
        """Devuelve los nombres resultantes para inspección y composición de pipelines."""
        check_is_fitted(self, attributes=["feature_names_out_"])
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features debe coincidir con las columnas usadas en fit.")
        return self.feature_names_out_.copy()

    @classmethod
    def _validate_input(cls, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer requiere un pandas.DataFrame.")
        missing = set(cls.required_columns).difference(X.columns)
        if missing:
            raise ValueError(
                f"Faltan columnas requeridas para crear features: {sorted(missing)}"
            )
