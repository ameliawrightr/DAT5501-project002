#ROLLING ORIGIN BACKTESTING

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_errors
from src.models.forecasting_utils import align_predictions

@dataclass
class BacktestConfig:
    """Configuration for rolling origin backtesting.

    Attributes:
        horizon: int
            Number of periods to forecast at each origin.
        initial_train_size: int
            Initial size of the training set.
        step_size: int
            Number of periods to move the origin forward at each step.
    """
    horizon: int = 1
    initial_train_size: int = 104
    step_size: int = 1

def _smape_point(y_true: float, y_pred: float) -> float:
    """Compute SMAPE for a single point."""
    denominator = (abs(y_true) + abs(y_pred)) / 2
    if denominator == 0:
        return 0.0
    return abs(y_true - y_pred) / denominator

def rolling_origin_backtest(
    y: pd.Series,
    forecaster: Callable[[pd.Series, int, Dict[str, Any]], pd.Series],
    config: Optional[BacktestConfig] = None,
    event_flags: Optional[pd.DataFrame] = None,
    model_name: str = "model",
    forecaster_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform rolling origin backtesting on a time series.

    Parameters:
        y: pd.Series
            The time series data to forecast.
        forecaster: Callable
            A forecasting function that takes in a training series, horizon, and additional kwargs.
        config: BacktestConfig, optional
            Configuration for the backtest. If None, default values are used.
        event_flags: pd.DataFrame, optional
            DataFrame containing event flags to be used as additional features.
        model_name: str
            Name of the forecasting model.
        forecaster_kwargs: Dict[str, Any], optional
            Additional keyword arguments to pass to the forecaster.
    
    Returns:
        detailed_results: pd.DataFrame
        aggregate_results: pd.DataFrame
    """
    if config is None:
        config = BacktestConfig()

    if forecaster_kwargs is None:
        forecaster_kwargs = {}

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex.")
    if y.index.has_duplicates:
        raise ValueError("y index contains duplicate timestamps.")
    
    #ensure time index is sorted
    y = y.sort_index()

    n_obs = len(y)
    horizon = int(config.horizon)
    initial_train_size = int(config.initial_train_size)
    step_size = int(config.step_size)

    if horizon <= 0:
        raise ValueError("config.horizon must be a positive integer.")
    if initial_train_size <= 0:
        raise ValueError("config.initial_train_size must be a positive integer.")
    if step_size <= 0:
        raise ValueError("config.step_size must be a positive integer.")

    if n_obs < initial_train_size + horizon:
        raise ValueError(
            f"Not enough observations ({n_obs}) for rolling-origin backtest with "
            f"initial_train_size={initial_train_size} and horizon={horizon}."
        )
    
    #prepare optional event flags
    if event_flags is not None:
        #align and ensure boolean
        if not isinstance(event_flags, pd.DataFrame):
            raise TypeError("event_flags must be a pandas DataFrame.")
        if not isinstance(event_flags.index, pd.DatetimeIndex):
            raise TypeError("event_flags must have a DatetimeIndex.")   
        event_flags = event_flags.reindex(y.index).fillna(False).astype(bool)

    detailed_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    #origin index: last index of training set for each iteration
    origin_end_positions = range(initial_train_size, n_obs - horizon + 1, step_size)

    for origin_end_pos in origin_end_positions:
        #training history is from 0 to origin_end_pos - 1
        y_train = y.iloc[:origin_end_pos]
        #forecast window from origin_end_pos to origin_end_pos + horizon - 1
        y_test = y.iloc[origin_end_pos: origin_end_pos + horizon]

        origin_time = y_train.index[-1]

        #call forecaster
        raw_forecast = forecaster(history=y_train, horizon=horizon, **forecaster_kwargs)

        if not isinstance(raw_forecast, pd.Series):
            raise TypeError(
                "Forecaster must return a pandas Series."
                f"Got {type(raw_forecast)} instead.")
        
        #align predictions to test index
        y_pred = align_predictions(raw_forecast, index=y_test.index)

        #aggregate metrics for this origin
        origin_errors = compute_errors(y_true=y_test, y_pred=y_pred)
        aggregate_rows.append(
            {
                "model": model_name,
                "origin_time": origin_time,
                "start_forecast_time": y_test.index[0],
                "end_forecast_time": y_test.index[-1],
                "n_points": len(y_test),
                "MAE": float(origin_errors["MAE"]),
                "RMSE": float(origin_errors["RMSE"]),
                "sMAPE": float(origin_errors["sMAPE"]),
            }
        )

        #detailed per timestamp errors
        for h_step, (ts, y_true_val, y_pred_val) in enumerate(
            zip(y_test.index, y_test.values, y_pred.values), start=1
        ):
            abs_error = float(abs(y_pred_val - y_true_val))
            sq_error = float((y_pred_val - y_true_val) ** 2)
            smape_val = float(_smape_point(y_true_val, y_pred_val))

            row: Dict[str, Any] = {
                "model": model_name,
                "origin_time": origin_time,
                "forecast_time": ts,
                "horizon_step": h_step,
                "y_true": float(y_true_val),
                "y_pred": float(y_pred_val),
                "abs_error": abs_error,
                "squared_error": sq_error,
                "sMAPE_point": smape_val,
            }

            #include event flags if provided
            if event_flags is not None:
                flags_row = event_flags.loc[ts]
                for col in event_flags.columns:
                    row[col] = bool(flags_row[col])
            detailed_rows.append(row)
    
    detailed_results = pd.DataFrame(detailed_rows)
    aggregate_results = pd.DataFrame(aggregate_rows)

    #sort results
    if not detailed_results.empty:
        detailed_results = detailed_results.sort_values(
            by=["forecast_time", "origin_time", "horizon_step"]
        ).reset_index(drop=True)
    
    if not aggregate_results.empty:
        aggregate_results = aggregate_results.sort_values(
            by=["origin_time"]
        ).reset_index(drop=True)

    return detailed_results, aggregate_results