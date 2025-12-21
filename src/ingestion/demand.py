#formalising files living under data/processed

from __future__ import annotations

import pandas as pd

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
    dm = pd.read_csv(demand_csv_path)
    dm["date"] = pd.to_datetime(dm["date"])
    dm["month"] = dm["date"].dt.to_period("M")

    #keep only relevant from monhtly
    dm = dm[["month", "category", "demand"]]
    
    #Load weekly calendar
    cal = pd.read_csv(
        calendar_csv_path,
        dayfirst=True,
        parse_dates=["week_start", "week_end"],
    )
    cal["month"] = cal["week_start"].dt.to_period("M")

    #Merge monthly demand onto weeks by (month, category)
    weekly_demand = cal.merge(dm, on=["month"], how="left")

    #convert monthly demand into weekly demand
    weekly_demand["weeks_in_month"] = weekly_demand.groupby(["month", "category"])["week_start"].transform("nunique")
    weekly_demand["demand"] = weekly_demand["demand"] / weekly_demand["weeks_in_month"]
    weekly_demand = weekly_demand.drop(columns=["weeks_in_month"])

    #detect event flag columns automatically
    event_cols = [c for c in weekly_demand.columns if c.startswith("is_")]
    for col in event_cols:
            weekly_demand[col] = weekly_demand[col].fillna(0).astype(bool)

    #return tidy weekly DF
    weekly_demand = weekly_demand.rename(columns={"week_start": "week_start"})
    weekly_demand = weekly_demand[["week_start", "category", "demand"] + event_cols]
    
    return weekly_demand