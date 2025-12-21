#EVENT AWARE FORECASTING MODELS
"""
This module implements autoregressive models with exogenous event features.
The aim is to test whether explicitly modelling event windows (New Year,
back-to-school, exam season, Q4 holidays, etc.) improves forecast accuracy
and stability compared to history-only baselines.

Design goals:
- No data leakage: models are trained only on past values.
- Forecasts generated recursively, 1-step-ahead over the horizon.
- Clear separation of:
    * feature engineering
    * model fitting
    * forecasting
so that unit tests can target each part independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np 
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Configuration + result containers
# --------------------------------------------------

ModelType = Literal["ridge", "random_forest"]

@dataclass
class EventModelConfig:
    """Config for event aware autoregressive models.
    
    Parameters
     - model_type:
        "ridge" (regularised linear regression) or "random_forest"
        (non-linear tree ensemble).
     - lags:
        Sequence of positive integer lags of the target variable to include
        as autoregressive features, e.g. (1, 52) for last week and
        same week last year.
     - include_trend:
        If True, include an integer time index feature.
     - include_weekofyear:
        If True, include the ISO week-of-year as a numeric feature.
     - alpha:
        Ridge regularisation strength (ignored for random forest).
     - rf_n_estimators:
        Number of trees for RandomForestRegressor.
     - rf_max_depth:
        Maximum depth of trees (None = unrestricted).
    """

    model_type: ModelType = "ridge"
    lags: Sequence[int] = (1, 52)
    include_trend: bool = True
    include_weekofyear: bool = True
    alpha: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None

@dataclass
class EventModelForecast:
    """Container for event model forecast results.
    
    Attributes
    - model:
        The trained scikit-learn regressor (Ridge or RandomForestRegressor).
    - config:
        Configuration used to build features and fit the model.
    -feature_columns:
        Names of input features used during training; used to ensure that
        forecast features are constructed consistently.
    - n_train_points:
        Number of points in the original training series (before lag cutting).
        Used to construct time-index features during forecasting.
    - max_lag:
        Maximum lag used in autoregressive features; used to determine how
    """

    model: RegressorMixin
    config: EventModelConfig
    feature_columns: List[str]
    n_train_points: int
    max_lag: int

# --------------------------------------------------
# Feature engineering helpers
# --------------------------------------------------

def _validate_alignment(
        y: pd.Series,
        events: pd.DataFrame
) -> None:
    #Raise if y and events are not aligned on same DateTimeIndex
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DateTimeIndex")
    if not isinstance(events.index, pd.DatetimeIndex):
        raise TypeError("events must have a DateTimeIndex")
    if not y.index.equals(events.index):
        raise ValueError(
            "y and events must be aligned on the same DateTimeIndex."
            f" Got {len(y.index)} y points and {len(events.index)} event rows."
        )
    
def build_autoreg_event_training_matrix(
        y: pd.Series,
        events: pd.DataFrame,
        config: EventModelConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct the training design matrix X and target vector y_target for an
    autoregressive model with event features.

    This function:
    - aligns y and events,
    - creates lag features y_{t-k} for each k in config.lags,
    - adds optional time-trend and week-of-year features,
    - drops the first max(lags) rows which do not have full lag history.

    Parameters
     - y:
        Training target series, indexed by week_start (DatetimeIndex).
     - events:
        DataFrame of boolean or numeric event indicators, aligned to y.
     - config:
        EventModelConfig specifying lags and optional time features.

    Returns
     - X :
        DataFrame of shape (n_samples, n_features).
     - y_target :
        Series of length n_samples aligned to X.
    """
    _validate_alignment(y, events)

    if len(y) != len(events):
        raise ValueError(
            "y and events must have the same length."
            f" Got {len(y)} y points and {len(events)} event rows."
        )
    
    #copy to avoid mutating original
    y = y.astype(float).copy()
    events = events.copy()

    max_lag = max(config.lags) if config.lags else 0
    if max_lag >= len(y):
        raise ValueError(
            "Max lag must be less than length of y."
            f" Got max_lag={max_lag} and len(y)={len(y)}."
        )
    
    feature_frames: List[pd.DataFrame] = []

    #1. Lag features of target variable
    for lag in config.lags:
        lagged = y.shift(lag)
        feature_frames.append(lagged.to_frame(name=f"lag_{lag}"))

    #2. Event indicator features
    event_features = events.copy()
    #ensure bool dtypes are converted to int (0/1)
    for col in event_features.columns:
        if event_features[col].dtype == bool:
            event_features[col] = event_features[col].astype(int)
    feature_frames.append(event_features)

    #3. Time trend feature
    if config.include_trend:
        time_index = np.arange(len(y), dtype=float)
        trend = pd.Series(
            time_index,
            index=y.index,
            name="time_index"
        )
        feature_frames.append(trend.to_frame())

    #4. Week-of-year feature
    if config.include_weekofyear:
        weekofyear = (
            y.index.isocalendar().week.astype(int).rename("week_of_year")
        )
        feature_frames.append(weekofyear.to_frame())

    #combine all features
    x_full = pd.concat(feature_frames, axis=1)

    #drop initial rows with incomplete lag history
    X = x_full.iloc[max_lag:].copy()
    y_target = y.iloc[max_lag:].copy()

    assert len(X) == len(y_target)

    return X, y_target

# --------------------------------------------------
# Model construction + training
# --------------------------------------------------

def _build_regressor(config: EventModelConfig) -> RegressorMixin:
    # Construct underlying scikit-learn regressor based on the config.
    if config.model_type == "ridge":
    #standardise numeric features for stable linear model
        model: RegressorMixin = Pipeline(
            steps = [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=config.alpha)),
            ]
        )
    elif config.model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            random_state=42
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {config.model_type!r}. "
            "Supported types are 'ridge' and 'random_forest'."
        )
    return model

def fit_event_model(
        y_train: pd.Series,
        events_train: pd.DataFrame,
        config: EventModelConfig | None = None,
) -> EventModelConfig:
    """Fit event aware autoregressive model to training data.
    
    Parameters:
     - y_train:
        Training target series, indexed by week_start (DatetimeIndex).
     - events_train:
        DataFrame of boolean or numeric event indicators, aligned to y_train.
     - config:
        EventModelConfig specifying lags and optional time features.
        If None, default config is used.
        
    Returns:
     - EventModelResult:
        Container with trained model and metadata for forecasting.
    """
    if config is None:
        config = EventModelConfig()

    #Build training design matrix
    X_train, y_target = build_autoreg_event_training_matrix(
        y=y_train,
        events=events_train,
        config=config
    )

    #Fit model
    model = _build_regressor(config)
    model.fit(X_train.values, y_target.values)

    result = EventModelForecast(
        model=model,
        config=config,
        feature_columns=list(X_train.columns),
        n_train_points=len(y_train),
        max_lag=max(config.lags) if config.lags else 0,
    )

    return result
    
#--------------------------------------------------
# Forecasting
#--------------------------------------------------

def _build_feature_row_for_forecast(
        history_window: Sequence[float],
        event_row: pd.Series,
        dt_index: pd.Timestamp,
        time_index_value: float,
        result: EventModelForecast
) -> np.ndarray:
    """Construct single feature row for forecasting at dt_index
    
    Parameters:
     - history_window:
        Sequence containing the last `max_lag` values of the target
        (actuals + previously predicted values).
     - event_row:
        Row from events_future with event indicators for time t.
     - dt_index:
        Timestamp for the forecasted week (for week-of-year feature).
     - time_index_value:
        Scalar time index used for the trend feature (e.g. position in series).
     - result:
        EventModelResult containing config and feature metadata.

    Returns
     - features:
        1D numpy array matching the ordering of result.feature_columns.
    """
    cfg = result.config
    max_lag = result.max_lag

    if max_lag > 0 and len(history_window) < max_lag:
        raise ValueError(
            f"history_window must have at least {max_lag} points."
        )
    
    features_dict: dict[str, float] = {}

    #1. Lag features 
    #history_window assumed to be ordered from oldest to most recent
    #take last max_lag entries as most recent history
    recent = list(history_window)[-max_lag:]
    for lag, value in zip(sorted(cfg.lags), reversed(recent)):
        #reversed (recent) so lag_1 = last value, lag_2 = second last, etc.
        features_dict[f"lag_{lag}"] = float(value)

    #2. Event indicator features
    for col in event_row.index:
        val = event_row[col]
        if isinstance(val, (bool, np.bool_)):
            val = int(val)
        features_dict[col] = float(val)

    #3. Time trend feature
    if cfg.include_trend:
        features_dict["time_index"] = float(time_index_value)

    #4. Week of year feature
    if cfg.include_weekofyear:
        week = int(dt_index.isocalendar().week)
        features_dict["week_of_year"] = float(week)

    #5. order according to feature_columns
    feature_values: List[float] = []
    for col in result.feature_columns:
        if col not in features_dict:
            raise ValueError(
                f"Feature column {col!r} not found in constructed features."
            )
        feature_values.append(features_dict[col])

    return np.array(feature_values, dtype=float)

def forecast_event_model(
        result: EventModelForecast,
        y_history: pd.Series,
        events_future: pd.DataFrame,
) -> pd.Series:
    """Generate multi-step forecasts using the trained event model.
    
    For each step in the forecast horizon:
    - features are built from the latest `max_lag` values of y_history plus
      any previously predicted values, and the corresponding row in
      events_future.
    - the model predicts demand for that week.
    - the prediction is appended to the history used for subsequent steps.

    Parameters
     - result:
        Fitted EventModelResult.
     - y_history:
        Series containing historical demand values up to the start of the
        forecast horizon. Its last values are used for lag features.
     - events_future:
        DataFrame of event indicators for the forecast horizon, with a
        DatetimeIndex giving the week_start for each forecast.

    Returns
     - forecasts :
        Series of length len(events_future) indexed by events_future.index.
    """
    cfg = result.config
    max_lag = result.max_lag

    if not isinstance(events_future.index, pd.DatetimeIndex):
        raise TypeError("events_future must have a DatetimeIndex.")

    #mutable list of history values
    history_values: List[float] = list(y_history.astype(float).values)

    preds: List[float] = []

    #time index continues from end of y_history
    base_time_index = float(result.n_train_points)

    for step, (dt, event_row) in enumerate(events_future.iterrows()):
        time_idx_value = base_time_index + step

        #build feature row
        features= _build_feature_row_for_forecast(
            history_window=history_values,
            event_row=event_row,
            dt_index=dt,
            time_index_value=time_idx_value,
            result=result
        )

        #reshape for sklearn (1, n_features)
        y_hat = float(result.model.predict(features.reshape(1, -1))[0])
        preds.append(y_hat)

        #append to history for next step lag
        history_values.append(y_hat)
    
    return pd.Series(preds, index=events_future.index, name="event_model_forecast")


# --------------------------------------------------
# Convenience wrapped for train + forecast on hold out test window
# --------------------------------------------------

def fit_and_forecast_event_model(
        y_train: pd.Series,
        y_test: pd.Series,
        events_train: pd.DataFrame,
        events_test: pd.DataFrame,
        config: EventModelConfig | None = None,
) -> Tuple[pd.Series, EventModelForecast]:
    """Convenience function for common case used in backtesting:
     - train on historical window and forecast over fixed hold out period

    Parameters:
        - y_train:
            Training target series, indexed by week_start (DatetimeIndex).
        - y_test:
            Test target series for forecast horizon, indexed by week_start.
        - events_train:
            Event indicators for training period (aligned to y_train).
        - events_test:
            Event indicators for test period (aligned to y_test).
        - config:
            EventModelConfig specifying lags and optional time features.
            If None, default config is used.
    
    Returns:
        - forecasts :
            Series of length len(y_test) indexed by y_test.index.
        - result :
            Fitted EventModelResult.
    """
    #sanity checks on alignment
    _validate_alignment(y_train, events_train)
    _validate_alignment(y_test, events_test)

    #fit on training portio
    result = fit_event_model(
        y_train=y_train,
        events_train=events_train,
        config=config
    )

    #generate forecasts for test window
    forecasts = forecast_event_model(
        result=result,
        y_history=y_train,
        events_future=events_test
    )

    forecasts = forecasts.reindex(y_test.index)

    return forecasts, result