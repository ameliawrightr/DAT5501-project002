#DEMAND INGESTION - ONS Retail Sales Data
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_csv
from .validation import (
    require_columns,
    require_unique_keys,
    require_non_null,
    require_numeric_range,
)

#--------------------------------------------------------------------------------------------
# INTERNAL HELPERS
#--------------------------------------------------------------------------------------------

#NEED TO CHECK THIS LIST IS COMPREHENSIVE ENOUGH <---
_DATE_COL_CANDIDATES = ["time_period","time period", "time", "period", "Time Period", "date", "month", "week", "index_date"]
_VALUE_COL_CANDIDATES = ["value", "v4_0", "obs_value", "observation", "index", "sales"]
#dont really get the relevance of value col candidates here

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

def _week_start_monday(dt: pd.Series) -> pd.Series:
    #convert to anchored on Monday, then start_time
    return dt.dt.to_period("W-MON").dt.start_time


#--------------------------------------------------------------------------------------------
# PUBLIC API
#--------------------------------------------------------------------------------------------


#1. validate time series assumptions and save processed file
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
    Read pre-cleaned monthly demand file (date, category, value),
    Outputs standardised monthly demand.

    Expected input columns:
     - date (monthly date)(datetime)
     - category (str)
     - volume_index (numeric)

    Output columns:
     - date (month start) (datetime)
     - category (str)
     - demand (numeric)
     """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input demand file not found: {in_path}")
    
    df = pd.read_csv(in_path)

    #validate required structure
    require_columns(df, [date_col, category_col, value_col], "demand_monthly_input")
    require_non_null(df, [date_col, category_col, value_col], "demand_monthly_input")

    #standardise
    df = df.rename(columns={value_col: "demand"}).copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    require_non_null(df, [date_col], "demand_monthly_parsed")

    #normalise to monthstart timestamp
    df["date"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    df["category"] = df[category_col].astype(str).str.strip()
    if category_remap:
        df["category"] = df["category"].replace(category_remap)

    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    require_non_null(df, ["demand"], "demand_monthly_numeric")

    #ensure one row per (month, category)
    df_out = df.groupby(["date", "category"], as_index=False)["demand"].mean()
    require_unique_keys(df_out, ["date", "category"], "demand_monthly")

    #write out
    out_path = Path(out_path)
    write_csv(df_out, out_path, dataset_name="demand_monthly")

    return df_out

