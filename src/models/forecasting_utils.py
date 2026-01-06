"""
Forecasting utility functions.

Provides small, reusable helpers for:
- constructing lag features from a univariate time series
- constructing simple calendar/time features from a DatetimeIndex
- aligning raw model predictions to a target DatetimeIndex
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd

#FEATURE ENGINEERING
#1. Create lag features for a time series
def make_lag_features(
    series: pd.Series,
    lags: Sequence[int]
) -> pd.DataFrame:
    """
    Each lag k creates a column 'lag_k' containing series.shift(k).

    Parameters:
    series : pd.Series
        The input time series.
    lags : Sequence[int]
        List of lag periods to create features for.

    Returns:
    pd.DataFrame
        DataFrame containing lag features aligned to original index.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pd.DatetimeIndex")
    
    lagged = {}

    for lag in lags:
        if lag <= 0:
            raise ValueError("All lags must be positive integers.")
        lagged[f"lag_{lag}"] = series.shift(lag)

    return pd.DataFrame(lagged, index=series.index)

#2. Create basic time based features from a DateTimeIndex
def make_time_features(
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Features include:
    - time_index : integer trend (0,1,2,...)
    - weekofyear : ISO week number (1-53)
    - month : calendar month (1-12)
    - year : calendar year (e.g., 2023)
    
    Parameters:
    index : pd.DatetimeIndex
        DateTimeIndex of the time series data.
        
    Returns:
    pd.DataFrame
        Feature matrix with same index.
    """
    if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Index must be a pd.DatetimeIndex")

    #ISO week information retruned as DF (year, week, day)
    iso = index.isocalendar()

    features = pd.DataFrame(
        {
                "time_index": np.arange(len(index), dtype=int),
                "weekofyear": iso.week.astype(int),
                "month": index.month.astype(int),
                "year": index.year.astype(int),
        },
        index=index
    )

    return features


#FORECAST ALIGNMENT
#3. Attach DateTimeIndex to sequence of predictions
def align_predictions(
    predictions: Sequence[float],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Use align baseline forecasts to test period - often computed wo DateTimeIndex.
    
    Parameters:
    - predictions : Sequence[float]
        Sequence of forecasted values.
    - index : pd.DatetimeIndex
        index to attach, len must = number of predictions.
        
    Returns:
    - pd.Series
        Forecast series with provided index
    
    Raises:
    - ValueError
        If length of predictions does not match length of index.
    """
    preds = np.asarray(predictions, dtype=float)

    if len(preds) != len(index):
         raise ValueError(
            f"Length mismatch: {len(preds)} predictions for "
            f"{len(index)} index values."
         )

    return pd.Series(preds, index=index)