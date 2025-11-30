"""
Simple ARIMA baseline model for LeBron's per-game points (PTS).

This script:
- Loads lebron_core_games.csv
- Builds a clean per-game time series with prepare_lebron_timeseries()
- Splits into time-based train/test
- Fits an ARIMA(p, d, q) model on PTS only (no exogenous features)
- Evaluates MAE and RMSE on the held-out test set
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------------------------------------------------
# Make sure we can import src/features/feature_engineering.py
# ----------------------------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from features.feature_engineering import prepare_lebron_timeseries  # type: ignore


# ----------------------------------------------------------------------
# 1. LOAD DATA AND BUILD PTS TIME SERIES
# ----------------------------------------------------------------------

def build_pts_series_from_csv(
    csv_path: str = "data/lebron_core_games.csv",
) -> pd.Series:
    """
    Load LeBron's core game logs from CSV and return a clean PTS time series.

    Returns
    -------
    y : pd.Series
        Index: GAME_DATE (DatetimeIndex)
        Values: PTS (points per game)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    core = pd.read_csv(csv_path)
    lebron_ts = prepare_lebron_timeseries(core)

    # Ensure datetime index and sorted
    if not isinstance(lebron_ts.index, pd.DatetimeIndex):
        lebron_ts.index = pd.to_datetime(lebron_ts.index)
    lebron_ts = lebron_ts.sort_index()

    # Target series
    y = lebron_ts["PTS"].astype(float)

    return y


# ----------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (TIME-AWARE)
# ----------------------------------------------------------------------

def time_based_train_test_split_series(
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time-indexed Series into train and test based on chronology.

    Parameters
    ----------
    y : pd.Series
        Full PTS series, indexed by GAME_DATE.
    test_size : float, default=0.2
        Fraction of the data to allocate to the test set (from the end).

    Returns
    -------
    y_train, y_test : (pd.Series, pd.Series)
    """
    y = y.sort_index()
    n = len(y)
    n_test = int(np.round(n * test_size))
    n_train = n - n_test

    y_train = y.iloc[:n_train].copy()
    y_test = y.iloc[n_train:].copy()

    return y_train, y_test


# ----------------------------------------------------------------------
# 3. FIT ARIMA MODEL (NO EXOG, IGNORE INDEX)
# ----------------------------------------------------------------------

def fit_arima(
    y_train: pd.Series,
    order: Tuple[int, int, int] = (1, 0, 1),
):
    """
    Fit a simple ARIMA model on the training series (PTS only).

    IMPORTANT: we pass y_train.values so ARIMA works on a plain array
    and doesn't try to be clever with the date index.
    """
    y_array = y_train.values.astype(float)

    model = ARIMA(
        endog=y_array,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    model_fit = model.fit()
    return model_fit


# ----------------------------------------------------------------------
# 4. EVALUATION HELPERS
# ----------------------------------------------------------------------

def evaluate_arima_forecast(
    model_fit,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Forecast over the test set and compute basic error metrics.

    We ignore the original index inside ARIMA (it was fit on an array),
    and just ask for `steps = len(y_test)` forecasts, then align them
    back to the y_test dates ourselves.
    """
    steps = len(y_test)
    forecast = model_fit.forecast(steps=steps)

    # Wrap predictions in a Series and align with y_test index
    y_pred = pd.Series(forecast, index=y_test.index, name="PTS_pred").astype(float)

    # Combine and drop any NaNs (should be very rare)
    df_eval = pd.concat(
        [y_test.rename("PTS_true"), y_pred],
        axis=1
    ).dropna()

    if df_eval.empty:
        raise ValueError("No valid forecast points: all y_true/y_pred are NaN.")

    y_true_clean = df_eval["PTS_true"]
    y_pred_clean = df_eval["PTS_pred"]

    mae = float(np.mean(np.abs(y_true_clean - y_pred_clean)))
    rmse = float(np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "y_true": y_true_clean,
        "y_pred": y_pred_clean,
    }


# ----------------------------------------------------------------------
# 5. HIGH-LEVEL TRAIN + EVAL PIPELINE
# ----------------------------------------------------------------------

def train_and_evaluate_arima(
    csv_path: str = "data/lebron_core_games.csv",
    test_size: float = 0.2,
    order: Tuple[int, int, int] = (1, 0, 1),
):
    """
    Full pipeline:
    - Build PTS time series from CSV.
    - Split into train / test by time.
    - Fit ARIMA model on training period.
    - Evaluate on held-out test period.
    - Print metrics and return artifacts.

    Returns
    -------
    model_fit : ARIMAResults
    metrics : dict
    y_train : pd.Series
    y_test : pd.Series
    """
    # 1) Build PTS series
    y = build_pts_series_from_csv(csv_path)

    # 2) Train/test split
    y_train, y_test = time_based_train_test_split_series(y, test_size=test_size)

    # 3) Fit model
    model_fit = fit_arima(y_train, order=order)

    # 4) Evaluate
    metrics = evaluate_arima_forecast(model_fit, y_train, y_test)

    print("=== ARIMA Evaluation on Test Set (PTS only) ===")
    print(f"Order (p, d, q): {order}")
    print(f"Test size: {test_size:.2f}  (n_test = {len(metrics['y_true'])})")
    print(f"MAE : {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")

    return model_fit, metrics, y_train, y_test


# ----------------------------------------------------------------------
# 6. SCRIPT ENTRY POINT (for quick testing)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Run a quick end-to-end training + evaluation when this file
    # is executed directly: `python src/models/arima_lebron_simple.py`
    model_fit, metrics, y_train, y_test = train_and_evaluate_arima()


