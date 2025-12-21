from __future__ import annotations

from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.demand import load_weekly_demand
from src.models.baseline import (
    seasonal_naive,
    rolling_average,
    time_regression,
)
from src.models.forecasting_utils import align_predictions
from src.evaluation.metrics import compute_errors

#1. 
def run_baselines_for_category(
        category: str,
        demand_csv_path: str = "data/processed/demand_monthly.csv",
        seasonal_period: int = 52,
        rolling_window: int = 12,
        test_weeks: int | None = 52,
) -> None:
    """Run baselines models for single product category and print metrics
    
    Parameters:
    - category: str
        Category to filter on
    - demand_csv_path: str
        Path to the demand CSV file
    - seasonal_period: int
        Seasonal period for seasonal naive model
    - rolling_window: int
        Rolling window size for rolling average model
    - test_weeks: int | None
        Number of weeks to test on (default: 52)
    """
    df = load_weekly_demand(demand_csv_path)

    df_cat = df[df["category"] == category].copy()
    if df_cat.empty:
        raise ValueError(f"No data found for category: {category}")
    
    df_cat = df_cat.set_index("week_start")

    #enforce weekly frequency,then fill missing weeks with 0 demand
    y = df_cat["demand"].asfreq("W-MON").fillna(0)

    if test_weeks is None:
        test_weeks = max(4, len(y) // 4)

    if len(y) <= test_weeks + seasonal_period:
        raise ValueError(
            f"Not enough data for category {category} to run baselines with "
            f"{test_weeks} test weeks and seasonal period {seasonal_period}."
        )
    
    y_train = y.iloc[:-test_weeks]
    y_test = y.iloc[-test_weeks:]
    horizon = len(y_test)

    print(f"\n Category: {category}")
    print(f"Train points: {len(y_train)}, Test points: {len(y_test)}")

    #Baseline 1: Seasonal Naive
    sn_raw = seasonal_naive(y_train, seasonal_period=seasonal_period, horizon=horizon)
    sn_forecast = align_predictions(sn_raw, index=y_test.index)
    sn_errors = compute_errors(y_test, sn_forecast)
    print("\n Seasonal Naive Errors:")
    pprint(sn_errors)

    #Baseline 2: Rolling Average
    ra_raw = rolling_average(y_train, window=rolling_window, horizon=horizon)
    ra_forecast = align_predictions(ra_raw, index=y_test.index)
    ra_errors = compute_errors(y_test, ra_forecast)
    print("\n Rolling Average Errors:")
    pprint(ra_errors)

    #Baseline 3: Time Regression
    y_train_clean = y_train.dropna()
    tr_forecast, _ = time_regression(
        history=y_train_clean, 
        horizon=horizon, 
        freq="W"
    )
    tr_forecast_aligned = tr_forecast.reindex(y_test.index)
    tr_errors = compute_errors(y_test, tr_forecast_aligned)
    print("\n Time Regression Errors:")
    pprint(tr_errors)

    #Quick plot of forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label="Train", color="blue")
    plt.plot(y_test.index, y_test, label="Test / Actual", color="black")
    plt.plot(y_test.index, sn_forecast, label="Seasonal Naive Forecast", color="orange")
    plt.plot(y_test.index, ra_forecast, label="Rolling Average Forecast", color="green")
    plt.plot(y_test.index, tr_forecast, label="Time Regression Forecast", color="red")
    plt.legend()
    plt.title(f"Baselines for Category: {category}")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main() -> None:
    categories = [
        "fitness_equipment",
        "school_supplies",
        "electronic_goods",
    ]

    for category in categories:
        run_baselines_for_category(category)

if __name__ == "__main__":
    main()