#CALENDAR + EVENTS INDICATOR INGESTION
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .validation import (
    require_columns,
    require_unique_keys,
    require_non_null,
)
from .io import write_csv

#1. Aggregate weekly event-indicator features into monthly event table
def build_events_monthly(
    in_path: Path,
    out_path: Path,
    *,
    week_start_col: str = "week_start",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Reads weekly event-feature CSV (one row per week) and produces a monthly 
    feature table (one row per month), where each monthly indicator is:
    
        monthly_flag = max(weekly_flag within month) 

    Appropriate when weekly event columns are binary indicators

    Output: data (month start) + flag columns
    """
    df = pd.read_csv(in_path)

    require_columns(df, [week_start_col], "events_weekly_input")

    #parse dd/mm/yyyy dates
    df[week_start_col] = pd.to_datetime(
        df[week_start_col],
        dayfirst=dayfirst,
        errors="coerce",
    )
    require_non_null(df, [week_start_col], "events_weekly_parsed")

    #Treat all non-date columns as event flags/features to be aggregated
    flag_cols = [c for c in df.columns if c != week_start_col]
    if not flag_cols:
        raise ValueError("No event flag columns found (expected columns other than week_start)")

    #Convert week_start to month start
    df["date"] = df[week_start_col].dt.to_period("M").dt.to_timestamp()

    #Aggregate by month using max()
    out = df.groupby("date", as_index=False)[flag_cols].max()
    #Enforce one row per month
    require_unique_keys(out, ["date"], "events_monthly")

    #Persist for downstream use
    write_csv(out, out_path, dataset_name="events_monthly")
    return out
