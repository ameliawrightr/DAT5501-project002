from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

#Paths to processed event files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WEEKLY_EVENTS_PATH = PROCESSED_DIR / "calendar_events_uk_weekly_1988_2025.csv"
MONTHLY_EVENTS_PATH = PROCESSED_DIR / "events_monthly.csv"

#--------------------------------------------------------------------
# Loaders
#--------------------------------------------------------------------
def load_weekly_event_calendar(
        path: Path = WEEKLY_EVENTS_PATH
) -> pd.DataFrame:
    #load weekly calendar/events features and parse dates
    events = pd.read_csv(path)

    #use DD/MM/YYYY date format
    events["week_start"] = pd.to_datetime(events["week_start"], dayfirst=True)
    events["week_end"] = pd.to_datetime(events["week_end"], dayfirst=True)

    return events

def load_monthly_events(
        path: Path = MONTHLY_EVENTS_PATH
) -> pd.DataFrame:
    #load monthly calendar/events features and parse dates
    events = pd.read_csv(path)

    #use DD/MM/YYYY date format
    events["event_date"] = pd.to_datetime(events["event_date"], dayfirst=True)

    return events

#--------------------------------------------------------------------
# Feature function
#--------------------------------------------------------------------

def add_weekly_event_features(
        demand_weekly: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        date_col: str = "week_start"
) -> pd.DataFrame:
    """ 
    Merge weekly calendar/event indicators onto weekly demand df
    
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

    #normalise to datetime for safe join
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in demand_weekly DataFrame.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    #align column name with events.week_start
    events_for_merge = events.rename(columns={"week_start": date_col})

    merged = pd.merge(
        events_for_merge,
        on=date_col,
        how="left",
        validate="m:1", #many demand rows to one event row
    )

    #sanity check - no missing event rows where demand
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
        date_col: str = "event_date"
) -> pd.DataFrame:
    """ 
    Merge monthly calendar/event indicators onto monthly demand df
    
    Parameters:
    - demand_monthly: pd.DataFrame
        DataFrame with monthly demand data, must contain date_col
    - events: Optional[pd.DataFrame]
        DataFrame with monthly event features, if None loads default
    - date_col: str
        Name of date column in demand_monthly to merge on (default "event_date")
    
    Returns:
    - pd.DataFrame
        demand_monthly with event features merged in
    """
    if events is None:
        events = load_monthly_events()

    df = demand_monthly.copy()

    #normalise to datetime for safe join
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in demand_monthly DataFrame.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    events_for_merge = events.rename(columns={"event_date": date_col})
    
    merged = df.merge(
        events_for_merge,
        on=date_col,
        how="left",
        validate="m:1", #many demand rows to one event row
    )

    if merged["is_new_year_fitness"].isna().any():
        missing = merged[merged["is_new_year_fitness"].isna()][[date_col]].head()
        raise ValueError(
            f"After merging, some rows in demand_monthly have no matching event data. "
            f"Examples: \n{missing}"
        )

    return merged