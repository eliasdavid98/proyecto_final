"""Preparación de datos y componentes de validación para churn."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.pipeline import make_model_pipeline

DEFAULT_HOLDOUT_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 3
TARGET_COLUMN = "exited"
EXCLUDED_COLUMNS = (
    TARGET_COLUMN,
    "observation_id",
    "customer_id",
    "customer_id_raw",
    "surname",
    "source_split",
)


@dataclass(frozen=True)
class ChurnDataSplit:
    """Partición reproducible de desarrollo, holdout y test de competencia."""

    X_development: pd.DataFrame
    X_holdout: pd.DataFrame
    y_development: pd.Series
    y_holdout: pd.Series
    X_competition_test: pd.DataFrame


def split_churn_data(
    obt: pd.DataFrame,
    holdout_size: float = DEFAULT_HOLDOUT_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ChurnDataSplit:
    """Separa train etiquetado en desarrollo/holdout y mantiene aparte test."""
    required = {"source_split", TARGET_COLUMN}
    missing = required.difference(obt.columns)
    if missing:
        raise ValueError(f"La OBT no contiene las columnas requeridas: {sorted(missing)}")
    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size debe estar entre 0 y 1, sin incluirlos.")

    source_split = obt["source_split"].astype("string").str.strip().str.lower()
    invalid_split = source_split.isna() | ~source_split.isin({"train", "test"})
    if invalid_split.any():
        raise ValueError("source_split solo puede contener 'train' o 'test'.")

    train_mask = source_split.eq("train")
    test_mask = source_split.eq("test")
    labeled_train = obt.loc[train_mask].copy()
    competition_test = obt.loc[test_mask].copy()
    if labeled_train.empty or competition_test.empty:
        raise ValueError("La OBT debe contener filas train y test.")
    if competition_test[TARGET_COLUMN].notna().any():
        raise ValueError("El split test de competencia debe permanecer sin etiqueta.")

    target = pd.to_numeric(labeled_train[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("El target Exited de train contiene valores nulos o inválidos.")
    classes = set(target.unique().tolist())
    if classes != {0, 1}:
        raise ValueError(f"Exited debe contener ambas clases binarias 0 y 1; recibido: {classes}.")
    if (target.value_counts() < 2).any():
        raise ValueError("Cada clase requiere al menos dos filas para crear el holdout.")
    target = target.astype("int8").rename(TARGET_COLUMN)

    X = labeled_train.drop(columns=list(EXCLUDED_COLUMNS), errors="ignore")
    X_competition_test = competition_test.drop(
        columns=list(EXCLUDED_COLUMNS), errors="ignore"
    )
    X_development, X_holdout, y_development, y_holdout = train_test_split(
        X,
        target,
        test_size=holdout_size,
        random_state=random_state,
        stratify=target,
    )
    return ChurnDataSplit(
        X_development=X_development,
        X_holdout=X_holdout,
        y_development=y_development,
        y_holdout=y_holdout,
        X_competition_test=X_competition_test,
    )


def make_stratified_cv(
    n_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> StratifiedKFold:
    """Crea CV estratificada y barajada para usar solo en el split de desarrollo."""
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


