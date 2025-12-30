#Run event aware forecasting models for each event driven product cateogry
"""This script:
 - Loads weekly demand and event indicators
 - Trains event aware auto regressive models
 - Evaluates on a 52 week hold out test window
 - Reports errors overall and restricted to event weeks
 - Generates diagnostic plots

To run: python -m src.pipeline.run_event_models
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.demand import load_weekly_demand
from src.evaluation.metrics import compute_errors
from src.models.event_models import (
    EventModelConfig,
    fit_and_forecast_event_model,
)

#--------------------------------------------
#Pretty print helpers 
#--------------------------------------------
def pretty_print_errors(model_name: str, errors: Dict[str, float]) -> None:
    #Pretty print error metrics.
    mae = float(errors["MAE"])
    rmse = float(errors["RMSE"])
    smape = float(errors["sMAPE"])

    print(
        f" {model_name:18s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | "
        f"sMAPE: {smape:6.2f}%"
    )

def print_event_window_errors(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    event_test: pd.DataFrame,
) -> None:
    """Compute and print errors restricted to event weeks.
     Assumes event_test contains boolean columns such as:
        - is_new_year_fitness
        - is_back_to_school
        - is_exam_season
        - is_q4_holiday_electronics
    """
    masks = {
        "Any event week": event_test.any(axis=1),
        "New Year Fitness": event_test.get(
            "is_new_year_fitness",
            pd.Series(False, index=event_test.index),
        ),
        "Back to School": event_test.get(
            "is_back_to_school",
            pd.Series(False, index=event_test.index),
        ),
        "Exam Season": event_test.get(
            "is_exam_season",
            pd.Series(False, index=event_test.index),
        ),
        "Q4 Holiday Electronics": event_test.get(
            "is_q4_holiday_electronics",
            pd.Series(False, index=event_test.index),
        ),
    }

    print(f"\n {model_name} - Errors on Event Weeks:")
    for label, mask in masks.items():
        mask = mask.reindex(y_true.index).fillna(False)
        if not mask.any():
            continue
    
        errors = compute_errors(y_true[mask], y_pred[mask])
        mae = float(errors["MAE"])
        rmse = float(errors["RMSE"])
        smape = float(errors["sMAPE"])
        print(
            f"  {label:16s} | MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | "
            f"sMAPE: {smape:6.1f}%"
        )


#--------------------------------------------
# Plotting
#--------------------------------------------
def plot_event_forecast(
        category: str,
        y_train: pd.Series,
        y_test: pd.Series,
        forecasts: Dict[str, pd.Series],
        train_weeks_to_show: int = 52,
) -> None:
    """Plot last `train_weeks_to_show` weeks of training data plus test window,
    overlaying forecasts from each event-aware model.

    Parameters
     - category:
        Category name used in the title and output filename.
     - y_train:
        Training demand series.
     - y_test:
        Test demand series.
     - forecasts:
        Mapping from model name -> forecast series (indexed like y_test).
     - train_weeks_to_show:
        Number of weeks of training history to display for context.
    """
    
    train_tail = y_train.iloc[-train_weeks_to_show:]

    #compute y lims from train + test
    y_min = min(train_tail.min(), y_test.min())
    y_max = max(train_tail.max(), y_test.max())
    padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    plt.figure(figsize=(12, 6))

    plt.plot(
        train_tail.index,
        train_tail.values,
        label="Train (last 52 weeks)",
        color="black",
        linewidth=1.5,
    )
    plt.plot(
        y_test.index,
        y_test.values,
        label="Test / Actual",
        color="gray",
        linestyle="dashed",
        linewidth=1.8,
    )

    #forecasts
    for name, pred in forecasts.items():
        plt.plot(
            pred.index,
            pred.values,
            label=f"Forecast - {name}",
            linewidth=1.5,
            linestyle="--",
        )
    
    plt.title(f"Event Model Forecasts - Category: {category} (weekly)")
    plt.xlabel("Week")
    plt.ylabel("Demand")
    plt.ylim(y_min, y_max)
    plt.legend(loc="upper left")
    plt.tight_layout()

    os.makedirs("artifacts/event_models", exist_ok=True)
    out_path = f"artifacts/event_models/event_models_{category}.png"
    plt.savefig(out_path)
    plt.close()


#--------------------------------------------
# Main pipeline
#--------------------------------------------

def run_event_models_pipeline(
        category: str,
        demand_csv_path: str = "data/processed/demand_monthly.csv",
        test_weeks: int = 52,
) -> None:
    """Run the event models pipeline for a specific category."""
    # Load and preprocess data
    df = load_weekly_demand(demand_csv_path)
    df_cat = df[df["category"] == category].copy()

    if df_cat.empty:
        raise ValueError(f"No data found for category: {category}")

    df_cat = df_cat.sort_values("week_start").set_index("week_start")

    #Target variable
    y = df_cat["demand"].astype(float).sort_index()

    #Event indicators
    event_cols = [c for c in df_cat.columns if c.startswith("is_")]
    if not event_cols:
        raise ValueError(
            f"No event indicator columns found for category: {category}"
        )
    df_events = df_cat[event_cols].asfreq("W-MON").fillna(False)

    if len(y) <= test_weeks + 60:
        #rough sanity check for enough data for lags 
        raise ValueError(
            f"Not enough data for category: {category!r} ."
            f"with test_weeks={test_weeks}, have {len(y)} weeks."
        )
    
    y_train = y.iloc[:-test_weeks]
    y_test = y.iloc[-test_weeks:]
    events_train = df_events.loc[y_train.index]
    events_test = df_events.loc[y_test.index]

    print(f"\nCategory: {category!r}")
    print(f"Train points: {len(y_train)}, Test points: {len(y_test)}")

    #config models to run
    model_configs: Dict[str, EventModelConfig] = {
        "Ridge (lags 1, 52)": EventModelConfig(
            model_type="ridge",
            lags=(1, 52),
            include_trend=True,
            include_weekofyear=True,
            alpha=1.0,
        ),
        "Random Forest (lags 1, 52)": EventModelConfig(
            model_type="random_forest",
            lags=(1, 52),
            include_trend=True,
            include_weekofyear=True,
            rf_n_estimators=300,
            rf_max_depth=None,
        ),
    }

    forecasts: Dict[str, pd.Series] = {}

    for name, config in model_configs.items():
        print(f"\nFitting event model: {name}")
        y_pred, _ = fit_and_forecast_event_model(
            y_train=y_train,
            y_test=y_test,
            events_train=events_train,
            events_test=events_test,
            config=config,
        )

        #overall errors
        errors = compute_errors(y_test, y_pred)
        print("\nOverall Test Errors:")
        pretty_print_errors(name, errors)

        #event week errors
        print("\nEvent Week Errors:")
        print_event_window_errors(name, y_test, y_pred, events_test)

        forecasts[name] = y_pred

    #plot all event model forecasts for this category
    plot_event_forecast(
        category=category,
        y_train=y_train,
        y_test=y_test,
        forecasts=forecasts,
        train_weeks_to_show=52,
    )

#--------------------------------------------
# Entry point
#--------------------------------------------
def main() -> None:
    categories = [
        "fitness_equipment",
        "school_supplies",
        "electronic_goods",
    ]

    for category in categories:
        run_event_models_pipeline(category)

if __name__ == "__main__":
    main()