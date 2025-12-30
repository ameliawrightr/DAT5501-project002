from __future__ import annotations
from typing import Dict, Any, Tuple

import pandas as pd
import os
import numpy as np 
import time

from src.models.backtest import (
    rolling_origin_backtest,
    BacktestConfig,
)
from src.models.baseline import (
    seasonal_naive,
    rolling_average,
    time_regression,
)
from src.models.event_models import (
    EventModelConfig,
    fit_event_model,
    forecast_event_model
)
from src.ingestion.demand import load_weekly_demand


#SMALL WRAPPERS SO SIGNATURES MATCH rolling_origin_backtest EXPECTATIONS
def sn_forecaster(history: pd.Series,horizon: int,seasonal_period: int,**kwargs) -> pd.Series:
    return seasonal_naive(history,horizon=horizon,seasonal_period=seasonal_period)

def ra_forecaster(history: pd.Series,horizon: int,window: int=12,**kwargs) -> pd.Series:
    return rolling_average(history,horizon=horizon,window=window)

def tr_forecaster(history: pd.Series,horizon: int,**kwargs) -> pd.Series:
    forecast, _ = time_regression(history=history,horizon=horizon)
    return forecast

def event_ridge_forecaster(
        history: pd.Series,
        horizon: int,
        events: pd.DataFrame,
        config: EventModelConfig | None = None,
        **kwargs
    ) -> pd.Series:
    if config is None:
        config = EventModelConfig(
            model_type="ridge",
            lags=(1,52),
            include_trend=True,
            include_weekofyear=True,
            alpha=1.0,
        )
    
    events = events.sort_index()

    #sanity: history index must be subset of events index
    if not history.index.isin(events.index).all():
        missing = history.index[~history.index.isin(events.index)]
        raise ValueError(f"Event DataFrame missing indices: {missing}")
    
    full_index = events.index
    last_time = history.index[-1]
    try:
        pos = full_index.get_loc(last_time)
    except KeyError:
        raise ValueError(f"Last timestamp of history {last_time} not found in events index")
    
    future_index = full_index[pos+1 : pos+1 + horizon]

    events_train = events.loc[history.index]
    events_future = events.loc[future_index]

    #fit on history + aligned events
    result = fit_event_model(
        y_train=history,
        events_train=events_train,
        config=config,
    )
    #forecast over future horizon
    forecast = forecast_event_model(
        result=result,
        y_history=history,
        events_future=events_future,
    )

    return forecast

def event_rf_forecaster(
        history: pd.Series,
        horizon: int,
        events: pd.DataFrame,
        config: EventModelConfig | None = None,
        **kwargs,
    ) -> pd.Series:
    if config is None:
        config = EventModelConfig(
            model_type="random_forest",
            lags=(1,52),
            include_trend=True,
            include_weekofyear=True,
            rf_n_estimators=300,
            rf_max_depth=12,
            rf_min_samples_leaf=5,
            rf_n_jobs=-1,
            rf_random_state=42,
        )
    
    events = events.sort_index()

    if not history.index.isin(events.index).all():
        missing = history.index[~history.index.isin(events.index)]
        raise ValueError(f"Event DataFrame missing indices: {missing}")
    
    full_index = events.index
    last_time = history.index[-1]
    try:
        pos = full_index.get_loc(last_time)
    except KeyError:
        raise ValueError(f"Last timestamp of history {last_time} not found in events index")
    
    future_index = full_index[pos+1 : pos+1 + horizon]

    events_train = events.loc[history.index]
    events_future = events.loc[future_index]

    t0= time.time()

    result = fit_event_model(
        y_train=history,
        events_train=events_train,
        config=config,
    )
    forecast = forecast_event_model(
        result=result,
        y_history=history,
        events_future=events_future,
    )
    print(f"[Timing] RF fit: {time.time() - t0:.2f} seconds | n={len(history)}")
    return forecast

#------------------------------
#Core backtest runner
#------------------------------

def run_backtests_for_category(
        category: str,
        demand_csv_path: str = "data/processed/demand_monthly.csv",
        horizon: int=4,
        initial_train_size: int=260, #5 years of weekly data as initial window
        step_size: int=4, #1 month step size
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Run rolling origin backtests for multiple forecasting models on a given product category.

    Parameters:
    - category: str
        The product category to run backtests for.
    - demand_csv_path: str
        Path to the CSV file containing demand data.
    - horizon: int
        Forecast horizon in weeks.
    - initial_train_size: int
        Initial training window size in weeks.
    - step_size: int
        Step size for rolling origin in weeks.

    Returns:
    - results : dict
        Mapping model_name -> (detailed_df, aggregate_df)    
    """
    print(f"\n=== Rolling origin backtests for category: {category} ===")

    df = load_weekly_demand(demand_csv_path)
    df_cat = df[df['category'] == category].copy()
    if df_cat.empty:
        raise ValueError(f"No data found for category: {category}")
    
    df_cat = df_cat.set_index("week_start")

    #demand series (weekly, fill missing weeks with 0)
    y = df_cat["demand"].astype(float).sort_index()

    #event flags (col starting with 'is_', resampled to weekly)
    event_cols = [col for col in df_cat.columns if col.startswith("is_")]
    if event_cols:
        event_flags = df_cat[event_cols].astype(bool).sort_index()
    else:
        event_flags = None

    config = BacktestConfig(
        horizon=horizon,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )

    models = {
        "seasonal_naive": {
            "forecaster": sn_forecaster,
            "kwargs": {"seasonal_period": 52},
        },
        "rolling_average": {
            "forecaster": ra_forecaster,
            "kwargs": {"window": 12},
        },
        "time_regression": {
            "forecaster": tr_forecaster,
            "kwargs": {},
        },
    }

    if event_flags is not None and not event_flags.empty:
        models.update(
            {
                "event_ridge": {
                    "forecaster": event_ridge_forecaster,
                    "kwargs": {"events": event_flags},
                },
                "event_random_forest": {
                    "forecaster": event_rf_forecaster,
                    "kwargs": {"events": event_flags},
                },
            }
        )

    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    artifacts_dir = os.path.join("artifacts", "backtests")
    os.makedirs(artifacts_dir, exist_ok=True)

    for model_name, spec in models.items():
        cfg = config
        if model_name == "event_random_forest":
            cfg = BacktestConfig(
                horizon=horizon,
                initial_train_size=initial_train_size, #need at least 2 years for RF lags
                step_size=12,
            )
        forecaster = spec["forecaster"]
        kwargs = spec.get("kwargs", {})

        print(f"\n[Info] Running backtest for model: {model_name}")

        try:
            t0 = time.time()
            detailed, aggregate = rolling_origin_backtest(
                y=y,
                forecaster=forecaster,
                config=cfg,
                event_flags=event_flags,
                model_name=model_name,
                forecaster_kwargs=kwargs,
            )
            dt = time.time() - t0
            print(f"[Timing] {model_name} backtest completed in {dt:.2f} seconds")
        except ValueError as e:
            print(f"[Error] Backtest for model {model_name} failed: {e}")
            continue

        #save to csv
        detailed_path = os.path.join(artifacts_dir, f"{category}_{model_name}_detailed.csv")
        aggregate_path = os.path.join(artifacts_dir, f"{category}_{model_name}_aggregate.csv")

        detailed.to_csv(detailed_path, index=False)
        aggregate.to_csv(aggregate_path, index=False)

        print(f"[Info] Saved detailed results to: {detailed_path}")
        print(f"[Info] Saved aggregate results to: {aggregate_path}")

        #Quick summary: avg errors across all origins
        if not aggregate.empty:
            mean_mae = aggregate["MAE"].mean()
            mean_rmse = aggregate["RMSE"].mean()
            mean_smape = aggregate["sMAPE"].mean()
            print(
                f"[SUMMARY] {category:18s} | {model_name:16s}"
                f" | mean MAE: {mean_mae:7.2f}"
                f" | mean RMSE: {mean_rmse:7.2f}"
                f" | mean sMAPE: {mean_smape:6.1f}%"
            )

        results[model_name] = (detailed, aggregate)

    return results


#Script entry point
def main() -> None:
    categories = [
        "fitness_equipment",
        "school_supplies",
        "electronic_goods",
       ]
    
    horizon = 4
    initial_train_size = 260
    step_size = 4

    all_results: Dict[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = {}

    for category in categories:
        cat_results = run_backtests_for_category(
            category=category,
            demand_csv_path="data/processed/demand_monthly.csv",
            horizon=horizon,
            initial_train_size=initial_train_size,
            step_size=step_size,
        )
        all_results[category] = cat_results

    print("\n=== Backtesting completed for all categories ===")

if __name__ == "__main__":
    main()