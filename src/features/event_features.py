from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

#Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WEEKLY_EVENTS_PATH = PROCESSED_DIR / "calendar_events_uk_weekly_1988_2025.csv"
MONTHLY_EVENTS_PATH = PROCESSED_DIR / "events_monthly.csv"

#--------------------------------------------------------------------
# Loaders
#--------------------------------------------------------------------
#1. Load weekly event calendar (UK) and parse date columns
def load_weekly_event_calendar(
        path: Path = WEEKLY_EVENTS_PATH
) -> pd.DataFrame:
    events = pd.read_csv(path)

    #use DD/MM/YYYY date format
    events["week_start"] = pd.to_datetime(events["week_start"], dayfirst=True)
    events["week_end"] = pd.to_datetime(events["week_end"], dayfirst=True)

    return events

#2. Load monthly event features and parse date column
def load_monthly_events(
        path: Path = MONTHLY_EVENTS_PATH
) -> pd.DataFrame:
    events = pd.read_csv(path)

    #use DD/MM/YYYY date format
    events["date"] = pd.to_datetime(events["date"], dayfirst=True)

    return events

#--------------------------------------------------------------------
# Feature merge helpers
#--------------------------------------------------------------------

def add_weekly_event_features(
        demand_weekly: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        date_col: str = "week_start"
) -> pd.DataFrame:
    """ 
    Merge weekly calendar/event indicators onto weekly demand df
    Merge performed as (many rows per week_start) --> events (one row per week_start)
    
    This function enforces:
      - `date_col` exists in `demand_weekly`
      - merge cardinality is many-to-one (m:1)
      - no demand rows are left without matching event features

    Parameters:
    - demand_weekly: pd.DataFrame
        DataFrame with weekly demand data, must contain date_col
    - events: Optional[pd.DataFrame]
        DataFrame with weekly event features, if None loads default
    - date_col: str
        Name of date column in demand_weekly to merge on (default "week_start")
    
    Returns:
    - pd.DataFrame
        demand_weekly with event features merged in
    """
    if events is None:
        events = load_weekly_event_calendar()

    df = demand_weekly.copy()

    #Fail fast if join key missing
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in demand_weekly DataFrame.")
    
    #Normalise join key to datetime to avoid "object vs datetime" issues
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    #Align events join key name to demand join key name
    events_for_merge = events.rename(columns={"week_start": date_col})

    #Merge event features onto demand. Many demand rows to one event row
    merged = pd.merge(
        events_for_merge,
        on=date_col,
        how="left",
        validate="m:1",
    )

    #Sanity check - no missing event rows where demand
    if merged["is_new_year_fitness"].isna().any():
        missing = merged[merged["is_new_year_fitness"].isna()][[date_col]].head()
        raise ValueError(
            f"After merging, some rows in demand_weekly have no matching event data. "
            f"Examples: \n{missing}"
        )
    return merged

def add_monthly_event_features(
        demand_monthly: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        date_col: str = "date"
) -> pd.DataFrame:
    """ 
    Merge monthly calendar/event indicators onto monthly demand df
    
    Merge performed as (many rows per date) --> events (one row per date)

    This function enforces:
      - `date_col` exists in `demand_monthly`
      - merge cardinality is many-to-one (m:1)
      - no demand rows are left without matching event features

    Parameters:
    - demand_monthly: pd.DataFrame
        DataFrame with monthly demand data, must contain date_col
    - events: Optional[pd.DataFrame]
        DataFrame with monthly event features, if None loads default
    - date_col: str
        Name of date column in demand_monthly to merge on (default "date")
    
    Returns:
    - pd.DataFrame
        demand_monthly with event features merged in
    """
    if events is None:
        events = load_monthly_events()

    df = demand_monthly.copy()

    #Fail fast if join key missing
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in demand_monthly DataFrame.")
    
    #Normalise join key to datetime for consistent joinng
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    #Align events join key name to demand join key name
    events_for_merge = events.rename(columns={"date": date_col})

    merged = df.merge(
        events_for_merge,
        on=date_col,
        how="left",
        validate="m:1", #many demand rows to one event row
    )

    #Sanity check - no missing event rows where demand
    if merged["is_new_year_fitness"].isna().any():
        missing = merged[merged["is_new_year_fitness"].isna()][[date_col]].head()
        raise ValueError(
            f"After merging, some rows in demand_monthly have no matching event data. "
            f"Examples: \n{missing}"
        )

    return merged