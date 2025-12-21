from __future__ import annotations

from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import os

from src.ingestion.demand import load_weekly_demand
from src.models.baseline import (
    seasonal_naive,
    rolling_average,
    time_regression,
)
from src.models.forecasting_utils import align_predictions
from src.evaluation.metrics import compute_errors

def pretty_print_errors(model_name: str, errors: dict) -> None:
    #Nice formatted print of errors
    mae = float(errors["MAE"])
    rmse = float(errors["RMSE"])
    smape = float(errors["sMAPE"])
    print(f" {model_name:18s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | sMAPE: {smape:6.1f}%")


#Compute and print errors restricted to event weeks
def print_event_window_errors(
        model_name: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        event_test: pd.DataFrame,
) -> None:
    #Compute and print errors restricted to event weeks
    masks = {
        "Any event week": event_test.any(axis=1),
        "New Year": event_test.get("is_new_year_fitness", pd.Series(False, index=event_test.index)),
        "Back to School": event_test.get("is_back_to_school", pd.Series(False, index=event_test.index)),
        "Exam Season": event_test.get("is_exam_season", pd.Series(False, index=event_test.index)),
        "Q4 Holiday": event_test.get("is_q4_holiday_electronics", pd.Series(False, index=event_test.index)),
    }

    print(f"\n {model_name} - Errors on Event Weeks:")
    for label, mask in masks.items():
        #ensure mask is Series aligned to y_true
        mask = mask.reindex(y_true.index).fillna(False)
        if not mask.any():
            continue

        errs = compute_errors(y_true[mask], y_pred[mask])
        mae = float(errs["MAE"])
        rmse = float(errs["RMSE"])
        smape = float(errs["sMAPE"])
        print(f" {label:16s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | sMAPE: {smape:6.1f}%")

"""WORK ON THIS"""
#Zoomed plot around last few years + test window so its readable
def plot_baselines_zoomed(
        category: str,
        y_train: pd.Series, 
        y_test: pd.Series, 
        sn_forecast: pd.Series, 
        ra_forecast: pd.Series, 
        tr_forecast: pd.Series
) -> None:
    
    plt.figure(figsize=(12, 6))

    #inc last 5 years of train for context (5*52=260 weeks)
    train_tail = y_train.iloc[-260:]
    plt.plot(train_tail.index, train_tail, label="Train (last 5 years)", color="blue")
    plt.plot(y_test.index, y_test, label="Test / Actual", color="black")

    plt.plot(y_test.index, sn_forecast, label="Seasonal Naive Forecast", color="orange")
    plt.plot(y_test.index, ra_forecast, label="Rolling Average Forecast", color="green")
    #plt.plot(y_test.index, tr_forecast, label="Time Regression Forecast", color="red")

    plt.title(f"Baseline Forecasts for {category} (weekly) - Zoomed Test Period")
    plt.xlabel("Week")
    plt.ylabel("Demand")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(f"artifacts/baselines/baselines_{category}.png", dpi=300)

    plt.show()


#Baseline backtest chart - how models would have performed on last year
"""Run baseline models for a single category and print metrics."""
#  - Train: full historical weekly demand to fit baselines (all weeks except last 52)
#  - Test: last 52 weeks of demand
#  - SN, RA, TR forecast for 52 week test period
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

    #event flags - keep event columns, resampled to weekly freq and filled
    event_cols = [col for col in df_cat.columns if col.startswith("is_")]
    df_events = df_cat[event_cols].asfreq("W-MON").fillna(False)

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

    #align event flags to train/test
    event_train = df_events.loc[y_train.index]
    event_test = df_events.loc[y_test.index]

    print(f"\nCategory: {category}")
    print(f"Train points: {len(y_train)}, Test points: {len(y_test)}")

    # ------ Baseline 1: Seasonal Naive ------
    sn_raw = seasonal_naive(y_train, seasonal_period=seasonal_period, horizon=horizon)
    sn_forecast = align_predictions(sn_raw, index=y_test.index)
    sn_errors = compute_errors(y_test, sn_forecast)
   

    # ------ Baseline 2: Rolling Average ------
    ra_raw = rolling_average(y_train, window=rolling_window, horizon=horizon)
    ra_forecast = align_predictions(ra_raw, index=y_test.index)
    ra_errors = compute_errors(y_test, ra_forecast)


    # ------ Baseline 3: Time Regression ------
    y_train_clean = y_train.dropna()
    tr_forecast, _ = time_regression(
        history=y_train_clean, 
        horizon=horizon, 
    )

    tr_forecast_aligned = tr_forecast.reindex(y_test.index)
    tr_errors = compute_errors(y_test, tr_forecast_aligned)
   

    # ------ Print Results ------
    """ Overall errors """
    print("\nOverall errors: ")
    pretty_print_errors("Seasonal Naive", sn_errors)
    pretty_print_errors("Rolling Average", ra_errors)
    pretty_print_errors("Time Regression", tr_errors)

    """ Event week errors """
    print("\nEvent week errors:")
    print_event_window_errors("Seasonal Naive", y_test, sn_forecast, event_test)
    print_event_window_errors("Rolling Average", y_test, ra_forecast, event_test)
    print_event_window_errors("Time Regression", y_test, tr_forecast_aligned, event_test)

    # ------ Debug / sanity check for events in test ------
    print("\n[DEBUG] event_test.head():")
    print(event_test.head())
    print("[DEBUG] Total event weeks in test set:", event_test.any(axis=1).sum())
    print("[DEBUG] event_test.sum():")
    print(event_test.sum())

    """ Old version
    # ------ Plot Zoomed Baseline Forecasts ------
    plot_baselines_zoomed(
        category=category,
        y_train=y_train,
        y_test=y_test,
        sn_forecast=sn_forecast,
        ra_forecast=ra_forecast,
        tr_forecast=tr_forecast_aligned,
    )
    """

    #return everything needed for plotting and downstream analysis
    return {
        "y_train": y_train,
        "y_test": y_test,
        "sn_forecast": sn_forecast,
        "ra_forecast": ra_forecast,
        "tr_forecast": tr_forecast_aligned,
    }

#Plot baseline forecasts for one category
#  - last "train_years_to_show" weeks of training data
#  - full test window
def plot_baselines_single_category(
        category: str,
        y_train: pd.Series, 
        y_test: pd.Series, 
        sn_forecast: pd.Series, 
        ra_forecast: pd.Series, 
        train_years_to_show: int = 52,
) -> None:
    
    # focus on last "train_years_to_show" weeks of training data
    train_tail = y_train.iloc[-train_years_to_show:]

    #compute y-lims based on train + test only
    y_min = min(train_tail.min(), y_test.min())
    y_max = max(train_tail.max(), y_test.max())
    padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    plt.figure(figsize=(12, 6))

    #plot training history (last year)
    plt.plot(train_tail.index, train_tail.values,
             label="Train (last year)", linewidth=1.5)
    
    #plot test / actuals
    plt.plot(y_test.index, y_test.values,
             label="Test / Actual", linewidth=1.5, linestyle="-")
    
    #plot baselines on test period only
    plt.plot(y_test.index, sn_forecast.values,
             label="Seasonal Naive Forecast", linewidth=1.5, linestyle="--")
    plt.plot(y_test.index, ra_forecast.values,
             label="Rolling Average Forecast", linewidth=1.5, linestyle="--")
    
    plt.title(f"Baseline Forecasts for {category} (weekly)")
    plt.xlabel("Week")
    plt.ylabel("Demand")
    plt.ylim(y_min, y_max)

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

#def run_time_reg_forecast

#Compare three event driven categories on one plot
def plot_category_comparison(demand_fitness,
                             demand_school,
                             demand_electronics,
                             start=None):
    if start is not None:
        demand_fitness = demand_fitness[demand_fitness.index >= start]
        demand_school = demand_school[demand_school.index >= start]
        demand_electronics = demand_electronics[demand_electronics.index >= start]

    plt.figure(figsize=(12, 6))
    plt.plot(demand_fitness.index, demand_fitness.values,
             label="Fitness Equipment", color="blue")
    plt.plot(demand_school.index, demand_school.values,
             label="School Supplies", color="orange")
    plt.plot(demand_electronics.index, demand_electronics.values,
             label="Electronic Goods", color="green")

    plt.title("Weekly Demand Comparison of Event-Driven Categories")
    plt.xlabel("Week")
    plt.ylabel("Demand")
    plt.legend(loc="upper left")
    plt.tight_layout()

    os.makedirs("artifacts/baselines", exist_ok=True)
    plt.savefig("artifacts/baselines/all_category_comparison.png", dpi=300)

    plt.show()

    
def main() -> None:
    categories = [
        "fitness_equipment",
        "school_supplies",
        "electronic_goods",
    ]

    for category in categories:
        results = run_baselines_for_category(category)
        plot_baselines_single_category(
            category,
            y_train=results["y_train"],
            y_test=results["y_test"],
            sn_forecast=results["sn_forecast"],
            ra_forecast=results["ra_forecast"],
        )
        
    
    plot_category_comparison(
        demand_fitness=load_weekly_demand().query("category == 'fitness_equipment'").set_index("week_start")["demand"],
        demand_school=load_weekly_demand().query("category == 'school_supplies'").set_index("week_start")["demand"],
        demand_electronics=load_weekly_demand().query("category == 'electronic_goods'").set_index("week_start")["demand"],
        start="2018-01-01",
    )

if __name__ == "__main__":
    main()


