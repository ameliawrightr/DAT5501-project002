"""
DEMAND INGESTION - ONS Retail Sales Data
This module standardises pre-cleaned demand extracts into modelling-ready datasets:

- Monthly demand: one row per (month_start, category), plus monthly event features.
- Weekly demand : one row per (week_start (Mon), category), plus weekly event features.

Assumptions:
- Input files are already filtered/pre-cleaned to contain the relevant categories and
  demand measure (e.g., volume index).
- Duplicate keys are resolved via mean aggregation before validation.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from .io import write_csv
from .validation import (
    require_columns,
    require_unique_keys,
    require_non_null,
)
from src.features.event_features import (
    add_weekly_event_features,
    add_monthly_event_features
)

#--------------------------------------------------------------------------------------------
# INTERNAL HELPERS
#--------------------------------------------------------------------------------------------

_DATE_COL_CANDIDATES = ["time_period","time period", "time", "period", "Time Period", "date", "month", "week", "index_date"]
_VALUE_COL_CANDIDATES = ["value", "v4_0", "obs_value", "observation", "index", "sales"]

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _normalise_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    #try find first candidate col name in df.columns
    #allow for normalisaiton (lowercase + underscores)
    cols = set(df.columns)
    for c in candidates:
        norm = _normalise_name(c)
        if norm in cols:
            return norm
    return None

def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    Tries robust parsing for common ONS time formats:
    - YYYY-MM
    - YYYY MMM
    - YYYY-MM-DD
    - datetime already
    """    
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    
    s2 = s.astype(str).str.strip()

    #direct parse
    parsed = pd.to_datetime(s2, errors="coerce", utc=False)

    #if lots of NaT forcing to first of month or YYYY-MM
    if parsed.isna().mean() > 0.2:
        parsed = pd.to_datetime(s2 + "-01", errors="coerce", utc=False)
    return parsed

def _is_long_format(df: pd.DataFrame) -> bool:
    #series identifier column and value column
    cols = set(df.columns)
    has_value = any(c in cols for c in _VALUE_COL_CANDIDATES)
    has_series = ("cdid" in cols) or ("series" in cols) or ("series_id" in cols) or ("item" in cols)
    date_col = _find_first_col(df, _DATE_COL_CANDIDATES)
    return bool(has_value and has_series and date_col)

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

#Anchor datetime series to week starting Monday
def _week_start_monday(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("W-MON").dt.start_time


#--------------------------------------------------------------------------------------------
# PUBLIC API
#--------------------------------------------------------------------------------------------

#1. Build standardised monthly demand table and append monthly event features
def build_demand_monthly(
    in_path: Path,
    out_path: Path,
    *,
    date_col: str = "date",
    category_col: str = "category",
    value_col: str = "volume_index",
    category_remap: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a standardised monthly demand table and append monthly event features.

    Expected input columns (pre-cleaned extract):
      - date         : month-like date column (string or datetime)
      - category     : category identifier
      - volume_index : numeric demand measure (or override via value_col)

    Output columns:
      - date     : month start timestamp (datetime64)
      - category : str
      - demand   : float
      - plus monthly event columns (e.g., is_new_year_fitness, ...)

    Processing steps:
      1) Read input CSV and validate required columns are present and non-null.
      2) Parse date and normalise to month start (YYYY-MM-01).
      3) Clean category labels (strip whitespace; optional remap).
      4) Coerce demand to numeric.
      5) Aggregate duplicates to a single row per (date, category) using mean.
      6) Merge in monthly event features using `add_monthly_event_features`.
      7) Write to CSV.
     """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input demand file not found: {in_path}")
    
    df = pd.read_csv(in_path)

    #validate input structure
    require_columns(df, [date_col, category_col, value_col], "demand_monthly_input")
    require_non_null(df, [date_col, category_col, value_col], "demand_monthly_input")

    #standardise column names
    df = df.rename(columns={value_col: "demand"}).copy()

    #parse dates and normalise to month start
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    require_non_null(df, [date_col], "demand_monthly_parsed")
    df["date"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    #clean and standardise category labels
    df["category"] = df[category_col].astype(str).str.strip()
    if category_remap:
        df["category"] = df["category"].replace(category_remap)

    #ensure demand is numeric
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    require_non_null(df, ["demand"], "demand_monthly_numeric")

    #enforce single row per (month, category)
    df_out = df.groupby(["date", "category"], as_index=False)["demand"].mean()
    require_unique_keys(df_out, ["date", "category"], "demand_monthly")

    #append monthly event features
    df_out = add_monthly_event_features(
        df_out,
        events=None, #helper loads events_monthly.csv
        date_col="date" #mathches normalised date col
    )

    out_path = Path(out_path)
    write_csv(df_out, out_path, dataset_name="demand_monthly")

    return df_out


#2. Build standardised weekly demand table and append weekly event features
def build_demand_weekly(
    in_path: Path,
    out_path: Path,
    *,        
    date_col: str = "date",
    category_col: str = "category",
    value_col: str = "volume_index",
    category_remap: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a standardised weekly demand table and append weekly event features.

    Expected input columns (pre-cleaned extract):
      - date         : week-like date column (string or datetime)
      - category     : category identifier
      - volume_index : numeric demand measure (or override via value_col)

    Output columns:
      - week_start / date : Monday week start timestamp (datetime64)
      - category          : str
      - demand            : float
      - plus weekly event columns (e.g., is_new_year_fitness, ...)

    Processing steps:
      1) Read input CSV and validate required columns are present and non-null.
      2) Parse date and normalise to Monday week-start timestamps.
      3) Clean category labels (strip whitespace; optional remap).
      4) Coerce demand to numeric.
      5) Aggregate duplicates to a single row per (date, category) using mean.
      6) Merge in weekly event features using `add_weekly_event_features`.
      7) Write to CSV.
     """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input demand file not found: {in_path}")
    
    df = pd.read_csv(in_path)

    require_columns(df, [date_col, category_col, value_col], "demand_weekly_input")
    require_non_null(df, [date_col, category_col, value_col], "demand_weekly_input")

    df = df.rename(columns={value_col: "demand"}).copy()

    #parse dates then normalise to week start (Monday)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    require_non_null(df, [date_col], "demand_weekly_parsed")
    df["date"] = _week_start_monday(df[date_col])

    df["category"] = df[category_col].astype(str).str.strip()
    if category_remap:
        df["category"] = df["category"].replace(category_remap)

    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    require_non_null(df, ["demand"], "demand_weekly_numeric")

    #ensure one row per (week, category)
    df_out = df.groupby(["date", "category"], as_index=False)["demand"].mean()
    require_unique_keys(df_out, ["date", "category"], "demand_weekly")

    #append weekly event features
    df_out = add_weekly_event_features(
        df_out,
        events=None, #helper loads calendar_events_uk_weekly_1988_2025.csv
        date_col="date" #mathches normalised date col
    )

    out_path = Path(out_path)
    write_csv(df_out, out_path, dataset_name="demand_weekly")

    return df_out