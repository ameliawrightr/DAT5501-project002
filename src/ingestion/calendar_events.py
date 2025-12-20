#CALENDAR + EVENTS INDICATOR INGESTION
from __future__ import annotations
from calendar import month
from pathlib import Path
import pandas as pd

from .validation import (
    ValidationError,
    require_columns,
    require_unique_keys,
    require_non_null,
    require_numeric_range,  
)
from .io import write_csv

#1. merge into single calendar events table and save
def build_events_monthly(
    in_path: Path,
    out_path: Path,
    *,
    week_start_col: str = "week_start",
    dayfirst: bool = True,
) -> pd.DataFrame:
    """
    Read prebuilt weekly events files, aggregates to monthly.
    Monthly flags = max(flag) across weeks in month
    Output: data (month start) + flag columns
    """
    df = pd.read_csv(in_path)

    require_columns(
        df,
        [week_start_col],
        "events_weekly_input",
    )


    #parse dd/mm/yyyy dates
    df[week_start_col] = pd.to_datetime(
        df[week_start_col],
        dayfirst=dayfirst,
        errors="coerce",
    )
    require_non_null(df, [week_start_col], "events_weekly_parsed")

    #all non date cols treated as flags
    flag_cols = [c for c in df.columns if c != week_start_col]
    if not flag_cols:
        raise ValueError("No event flag columns found (expected columns other than week_start)")
    
    df["date"] = df[week_start_col].dt.to_period("M").dt.to_timestamp()

    out = df.groupby("date", as_index=False)[flag_cols].max()
    require_unique_keys(out, ["date"], "events_monthly")

    write_csv(out, out_path, dataset_name="events_monthly")
    return out
