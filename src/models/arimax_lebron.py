"""
ARIMAX modeling for LeBron James game-by-game points.

This module:
- Loads LeBron's per-game core stats from CSV.
- Uses the feature engineering pipeline to build `lebron_features`.
- Performs a time-based train/test split (no shuffling).
- Fits an ARIMAX model (SARIMAX with exogenous regressors).
- Evaluates performance on the test set (MAE, RMSE).
- Exposes helper functions so we can later plug in
  "predict next game vs X" style functionality.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ----------------------------------------------------------------------
# 1. IMPORT FEATURE PIPELINE
# ----------------------------------------------------------------------
# Adjust this import path depending on where your file actually lives.
# If `data_prep_feat_eng.py` is alongside this file, this is fine.
# If it's in src/features/, you might need:
# from src.features.data_prep_feat_eng import prepare_lebron_timeseries, build_lebron_feature_table

from src.features.feature_engineering import (
    prepare_lebron_timeseries,
    build_lebron_feature_table,
)


# ----------------------------------------------------------------------
# 2. BUILD FEATURE TABLE FROM CSV
# ----------------------------------------------------------------------

def build_features_from_csv(
    csv_path: str = "data/lebron_core_games.csv",
) -> pd.DataFrame:
    """
    Load LeBron's core game logs from CSV and build the full feature table
    using the existing feature engineering pipeline.

    Returns
    -------
    lebron_features : pd.DataFrame
        Index: GAME_DATE
        Columns: target (PTS) + all engineered exogenous features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    core = pd.read_csv(csv_path)
    lebron_ts = prepare_lebron_timeseries(core)
    lebron_features = build_lebron_feature_table(lebron_ts)

    return lebron_features


# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (TIME-AWARE)
# ----------------------------------------------------------------------

def time_based_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into train and test based on chronology.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature table (indexed by GAME_DATE).
    test_size : float, default=0.2
        Fraction of the data to allocate to the test set (from the end).

    Returns
    -------
    train_df, test_df : (pd.DataFrame, pd.DataFrame)
    """
    n = len(df)
    n_test = int(np.round(n * test_size))
    n_train = n - n_test

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    return train_df, test_df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate target (y = PTS) and exogenous features (X) for ARIMAX.

    - y: LeBron's points per game.
    - X: all numeric columns except PTS (avoids MATCHUP, OPP_TEAM_ABBR, etc.).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if "PTS" not in df.columns:
        raise KeyError("Expected 'PTS' column in feature table.")

    y = df["PTS"]

    # Only numeric predictors, drop target
    X = df.select_dtypes(include=["number"]).drop(columns=["PTS"])

    return X, y


# ----------------------------------------------------------------------
# 4. FIT ARIMAX MODEL
# ----------------------------------------------------------------------

def fit_arimax(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    order: Tuple[int, int, int] = (2, 0, 2),
):
    """
    Fit an ARIMAX model (SARIMAX with exogenous regressors) on the training set.

    Parameters
    ----------
    y_train : pd.Series
        Target time series (PTS).
    X_train : pd.DataFrame
        Exogenous regressors aligned with y_train.
    order : tuple, default=(2, 0, 2)
        ARIMA(p, d, q) order.

    Returns
    -------
    model_fit : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Fitted ARIMAX model.
    """
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    model_fit = model.fit(disp=False)
    return model_fit


# ----------------------------------------------------------------------
# 5. EVALUATION HELPERS
# ----------------------------------------------------------------------

def evaluate_forecast(
    model_fit,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Forecast over the test set and compute basic error metrics.

    Parameters
    ----------
    model_fit : SARIMAXResults
        Fitted ARIMAX model.
    X_test : pd.DataFrame
        Exogenous regressors for the test period.
    y_test : pd.Series
        True PTS values for the test period.

    Returns
    -------
    metrics : dict
        Contains MAE, RMSE, and a pd.Series of predictions.
    """
    # Forecast the entire test window given exogenous test features
    forecast_res = model_fit.get_forecast(
        steps=len(y_test),
        exog=X_test,
    )
    y_pred = forecast_res.predicted_mean

    # Align indices just in case
    y_pred = pd.Series(y_pred, index=y_test.index, name="PTS_pred")

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
    }


# ----------------------------------------------------------------------
# 6. HIGH-LEVEL TRAIN + EVAL PIPELINE
# ----------------------------------------------------------------------

def train_and_evaluate_arimax(
    csv_path: str = "data/lebron_core_games.csv",
    test_size: float = 0.2,
    order: Tuple[int, int, int] = (2, 0, 2),
):
    """
    Full pipeline:
    - Build feature table from CSV.
    - Split into train / test by time.
    - Fit ARIMAX model on training period.
    - Evaluate on held-out test period.
    - Print metrics and return artifacts.

    Returns
    -------
    model_fit : SARIMAXResults
    metrics : dict
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    # 1) Build feature table
    lebron_features = build_features_from_csv(csv_path)

    # 2) Train/test split
    train_df, test_df = time_based_train_test_split(lebron_features, test_size=test_size)

    # 3) Get X, y
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    # 4) Fit model
    model_fit = fit_arimax(y_train, X_train, order=order)

    # 5) Evaluate
    metrics = evaluate_forecast(model_fit, X_test, y_test)

    print("=== ARIMAX Evaluation on Test Set ===")
    print(f"Order (p, d, q): {order}")
    print(f"Test size: {test_size:.2f}  (n_test = {len(y_test)})")
    print(f"MAE : {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")

    return model_fit, metrics, train_df, test_df


# ----------------------------------------------------------------------
# 7. SCRIPT ENTRY POINT (for quick testing)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Run a quick end-to-end training + evaluation when this file
    # is executed directly: `python arimax_lebron.py`
    model_fit, metrics, train_df, test_df = train_and_evaluate_arimax()
