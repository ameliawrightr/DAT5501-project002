#BASELINE FORCECASTING MODELS
"""
For event driven retail demand

Provides simple baseline forecasting models:
1. Seasonal Naive
2. Rolling Mean
3. Time Only Linear Regression

Baselines used as reference points for more complex, event-aware models.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.forecasting_utils import (
    make_lag_features, 
    make_time_features, 
    align_predictions
)

#1. Seasonal Naive Model
def seasonal_naive(
        history: pd.Series,
        seasonal_period: int,
        horizon: int,
) -> pd.Series:
    """Seasonal Naive Forecasting Model

    Repeats last observed seasonal cycle for forecast horizon.

    Parameters:
    - history: pd.Series
        Historical time series data, indexed by date.
    - seasonal_period: int
        Number of periods in one seasonal cycle (e.g., 7 for weekly seasonality).
    - horizon: int
        Number of periods to forecast.

    Returns:
    - forecast: pd.Series
        Forecasted values for the specified horizon.
    """
    if len(history) < seasonal_period:
        raise ValueError(
            f"Need at least {seasonal_period} observations for "
            f" seasonal naive, got {len(history)}."
        )
    
    last_season = history.iloc[-seasonal_period:].to_numpy()

    #repeat last season as many times as needed to cover horizon
    repeats = int(np.ceil(horizon / seasonal_period))
    tiled = np.tile(last_season, repeats=repeats)
    forecast_values = tiled[:horizon]

    return pd.Series(forecast_values)

#2. Rolling Mean Model
def rolling_average(
        history: pd.Series,
        window: int, 
        horizon: int,
) -> pd.Series:
    """Rolling Average Forecasting Model

    Uses the rolling mean of the historical data as the forecast.

    Parameters:
    - history: pd.Series
        Historical time series data, indexed by date.
    - window: int
        Window size for calculating the rolling mean.
    - horizon: int
        Number of periods to forecast.
    
    Returns:
    - pd.Series
        Forecasted values for the specified horizon.
    """

    if len(history) < window:
        raise ValueError(
            f"Need at least {window} observations for "
            f"rolling average, got {len(history)}."
        )

    mean_value = history.iloc[-window:].mean()
    forecast_values = np.full(shape=horizon, fill_value=mean_value, dtype=np.float)

    return pd.Series(forecast_values)

#3. Time Only Linear Regression Model
def time_regression(
        history: pd.Series,
        horizon: int,
        freq: str | None = None,
) -> Tuple[pd.Series, LinearRegression]:
    """Time Only Linear Regression Forecasting Model

    Uses time-based features to predict future values via linear regression.

    Parameters:
    - history: pd.Series
        Historical time series data, indexed by date.
    - horizon: int
        Number of periods to forecast.
    - freq: str, optional
        Pandas offset alias for frequency of the time series (e.g., 'D' for daily).
        If None, inferred from history index.

    Returns:
    - forecast :pd.Series
        Forecasted values for the specified horizon.
        Indexed by future DataTimeIndex.
    - model : LinearRegression
        Trained linear regression model.
        Fitted scikit-learn LinearRegression model.

    Raises:
    - ValueError
        If index is not DateTimeIndex or frequency cannot be inferred.
    """
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("History index must be a DateTimeIndex.")
    
    #infer frequency if not provided
    if freq is None:
        inferred = history.index.freq or pd.infer_freq(history.index)
        if inferred is None:
            raise ValueError(
                "Could not infer frequency from history index. "
                "Please provide the 'freq' parameter."
            )
        freq = inferred
    
    #build training feature matrix from existing dates
    X_train = make_time_features(history.index)
    y_train = history.to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    #create future index
    last_timestamp = history.index[-1]
    future_index = pd.date_range(
        start=last_timestamp + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq
    )

    #create future feature matrix
    X_future = make_time_features(future_index)
    y_pred = model.predict(X_future)

    #generate forecasts
    forecast = pd.Series(y_pred, index=future_index)

    return forecast, model