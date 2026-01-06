""" 
Weekly demand proxy construction (data/processed)

This module builds a *weekly* demand proxy by disaggregating monthly demand across the
weeks in each month and attaching weekly calendar/event indicators.

Key idea:
- Demand is observed monthly (by category).
- Calendar/event indicators are observed weekly.
- We map each week to its calendar month, join monthly demand by (month, category),
  then split the monthly demand evenly across all weeks that fall in that month.

Important:
- This is a proxy. Equal-split disaggregation is a simplifying assumption and can
  distort intra-month dynamics (especially around events).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# ------------------------------------
#Internal validation helpers
# ------------------------------------
#1. Validate weekly proxy integrity
def _validate_weekly_proxy(df: pd.DataFrame) -> None:
    """
    Enforces:
    - required columns present
    - no nulls in key columns
    - week_start is datetime and always a Monday
    - each category has complete set of weeks
    """
    required_columns = {"week_start", "category", "demand"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Weekly proxy missing required columns: {sorted(missing)}")
    
    #null checks
    if df["week_start"].isna().any():
         raise ValueError("week_start contains null values")
    if df["category"].isna().any():
         raise ValueError("category contains null values")
    if df["demand"].isna().any():
         raise ValueError("demand contains null values after merge/disaggregation")

    #week_start must be datetime for time-based operations and plotting
    if not pd.api.types.is_datetime64_any_dtype(df["week_start"]):
        raise ValueError("week_start must be datetime type")
    
    #all weeks should start on a Monday
    weekdays = df["week_start"].dt.weekday.unique()
    if len(weekdays) != 1 or weekdays[0] != 0:
        raise ValueError(f"week_start must always be a Monday, got weekdays={weekdays}")
    
    #each category should have same number of unique week_start values
    counts = df.groupby("category")["week_start"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"Uneven week coverage by category: {counts.to_dict()} ")


# ------------------------------------
#Main function: load weekly demand proxy
# ------------------------------------
def load_weekly_demand(
        demand_csv_path: str = "data/processed/demand_monthly.csv",
        calendar_csv_path: str = "data/processed/calendar_events_uk_weekly_1988_2025.csv",
    ) -> pd.DataFrame:
    """Build weekly demand with event flags from monthly demand + weekly calendar.

    Inputs:
    - demand_monthly.csv: columns include
        ['date', 'category', 'demand', 'is_new_year_fitness',
         'is_back_to_school', 'is_exam_season', 'is_q4_holiday_electronics', ...]
    - calendar_events_uk_weekly_1988_2025.csv: columns include
        ['week_start', 'week_end', 'year', 'week_of_year',
         'is_new_year_fitness', 'is_back_to_school', 'is_exam_season',
         'is_q4_holiday_electronics', 'weeks_to_*', ...]

    We:
    - map each week to its calendar month,
    - join monthly demand onto weeks by (month, category),
    - split monthly demand evenly across the weeks of that month.
    """
    
    #Load monthly demand
    dm = pd.read_csv(demand_csv_path, parse_dates=["date"])
    if dm.empty:
        raise ValueError(f"Monthly demand file is empty: {demand_csv_path}")
    
    for col in ["date", "category", "demand"]:
        if col not in dm.columns:
            raise ValueError(f"Monthly demand missing required column: {col}")
        
    dm = dm.copy()
    dm["month"] = dm["date"].dt.to_period("M")

    #Only keep cols needed for disaggregation
    dm = dm[["month", "category", "demand"]].sort_values(["month", "category"])

    #Load weekly calendar
    cal = pd.read_csv(
        calendar_csv_path,
        dayfirst=True,
        parse_dates=["week_start", "week_end"],
    )
    if cal.empty:
        raise ValueError(f"Weekly calendar file is empty: {calendar_csv_path}")
    
    if "week_start" not in cal.columns:
        raise ValueError("Weekly calendar missing required column: week_start")
    
    cal = cal.copy()
    cal["month"] = cal["week_start"].dt.to_period("M")

    #detect event flag columns from calendar (weekly)
    event_cols = [c for c in cal.columns if c.startswith("is_")]

    #expand calendar to all categories (robust join key)
    categories = sorted(dm["category"].unique())
    cal_expanded = cal.assign(_key=1).merge(
        pd.DataFrame({"category": categories, "_key": 1}),
        on="_key",
        how="left",
    ).drop(columns=["_key"])

    #Merge monthly demand onto (week, category)
    weekly = cal_expanded.merge(dm, on=["month", "category"], how="left")

    if weekly["demand"].isna().any():
        #missing momthly demand for some (week, category)
        bad = weekly.loc[weekly["demand"].isna(), ["month", "category"]].drop_duplicates()
        raise ValueError(
            f"Missing monthly demand for some (month, category)."
            f"Example missing keys:\n{bad.head(10)}"
        )

    #disagg monthly -> weekly proxy
    weeks_in_month = weekly.groupby(["month", "category"])["week_start"].transform("nunique")
    weekly["demand"] = weekly["demand"] / weeks_in_month

    #clean event flags
    #event flags come from calendar; ensure boolean
    for col in event_cols:
        weekly[col] = weekly[col].fillna(0).astype(bool)

    output_cols = ["week_start", "category", "demand"] + event_cols
    weekly = weekly[output_cols].sort_values(["week_start", "category"]).reset_index(drop=True)

    _validate_weekly_proxy(weekly)
    return weekly
