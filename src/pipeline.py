"""Pipeline sklearn que encapsula ingeniería y preprocesamiento de churn."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import ChurnFeatureEngineer


def make_preprocessor() -> ColumnTransformer:
    """Construye imputación, escalado y one-hot para ajustar dentro de cada fold."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            (
                "categorical",
                categorical,
                make_column_selector(
                    dtype_include=["object", "string", "category"]
                ),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Mantiene feature engineering y preprocesamiento dentro del estimador CV."""
    return Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessing", make_preprocessor()),
            ("model", estimator),
        ]
    )
