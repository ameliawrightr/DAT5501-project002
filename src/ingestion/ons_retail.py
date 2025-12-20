#DEMAND INGESTION - ONS Retail Sales Data
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_csv_or_excel, write_parquet
from .validation import (
    require_columns,
    require_unique_keys,
    require_no_missing_weeks,
    require_non_null,
    require_numeric_range,
)

#--------------------------------------------------------------------------------------------
# INTERNAL HELPERS
#--------------------------------------------------------------------------------------------

#NEED TO CHECK THIS LIST IS COMPREHENSIVE ENOUGH <---
_DATE_COL_CANDIDATES = ["Time Period","date", "time", "period", "month", "week", "index_date"]
_VALUE_COL_CANDIDATES = ["value", "v4_0", "obs_value", "observation", "index", "sales"]
#dont really get the relevance of value col candidates here

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
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

#1. load raw ONS file
def load_ons_retail_raw(
        path: Path, 
        *, 
        sheet_name: str | None = None, 
        **kwargs: Any
    ) -> pd.DataFrame:
    """
    Read raw ONS retail data file into dataframe.
     - Does not reshape to tidy format
     - Does minimal column normalisation
     
    Parameters:
     - path: Path
        CSV/Excel file path
     - sheet_name : str | None
        optional sheet name for Excel files
     - kwargs : Any
        additional arguments passed to the read function

    Returns:
     - pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"ONS retail data file not found: {path}")

    df = read_csv_or_excel(path, sheet_name=sheet_name, **kwargs)
    df = _normalise_cols(df)

    if df.empty:
        raise ValueError(f"ONS raw dataframe empty after loading: {path}")
    
    return df
                                   
#2. map ONS series to 3 categories
def map_ons_to_categories(
        df_raw: pd.DataFrame, 
        mapping: dict[str, str]
    ) -> pd.DataFrame:
    """
    1. Map ONS series to project categories
    2. Return tidy table
    
    Parameters:
     - df_raw : pd.DataFrame
        Raw ONS retail dataframe
     - mapping : dict[str, str]
        Mapping from ONS series id to project category
        Dict of {ons_series_id: project_category}
         - if df WIDE format, series id is column names
         - if df LONG format, series id is in 'series' or 'cdid' column
    
    Returns:
     - pd.DataFrame
        Tidy dataframe with columns: 
         - date (datetime)
         - category (str)
         - value (float)
    """
    if not mapping:
        raise ValueError("Mapping dictionary is empty.")
    
    df = _normalise_cols(df_raw)

    date_col = _find_first_col(df, _DATE_COL_CANDIDATES)
    if date_col is None:
        raise ValueError(
            "Could not find a date column in raw data."
            f"Tried: {list(_DATE_COL_CANDIDATES)}"
        )

    #LONG FORMAT PATH
    if _is_long_format(df):
        #identify series ID column
        series_col = None
        for candidate in ["cdid", "series", "series_id", "item"]:
            if candidate in df.columns:
                series_col = candidate
                break
        if series_col is None:
            raise ValueError("Could not find series ID column in long format data.")
        
        value_col = _find_first_col(df, _VALUE_COL_CANDIDATES)
        if value_col is None:
            raise ValueError("Could not find value column in long format data.")
        
        out = df[[date_col, series_col, value_col]].copy()
        out.rename(
            columns={
                date_col: "date",
                series_col: "series_key",
                value_col: "value"
            },
            inplace=True
        )

        out["date"] = _parse_date_series(out["date"])
        out["value"] = _coerce_numeric(out["value"])

        #normalise keys for matching
        map_norm = {str(k).strip().lower(): v for k, v in mapping.items()}
        out["series_key_norm"] = out["series_key"].astype(str).str.strip().lower()
        out["category"] = out["series_key_norm"].map(map_norm)

        out = out[out["series_key_norm"].isin(map_norm.keys())].copy()
        if out.empty:
            raise ValueError(
                "No rows matched your mapping keys in long format data."
                "Check mapping keys vs series identifier column."
            )
        
        out["category"] = out["series_key_norm"].map(map_norm)
        out.drop(columns=["series_key", "series_key_norm"], inplace=True)

        require_columns(out, ["date", "category", "value"], "ons_tidy_long")
        require_non_null(out, ["date", "category"], "ons_tidy_long")

        return out[["date", "category", "value"]]

    #WIDE FORMAT PATH
    df["date"] = _parse_date_series(df[date_col])
    wide_cols = [c for c in df.columns if c != date_col]

    #mapping keys must correspond to wide column names
    map_norm = {str(k).strip().lower().replace(" ", "_"): v for k, v in mapping.items()}
    available = set(df.columns)

    missing = [k for k in map_norm.keys() if k not in available]
    if missing:
        raise ValueError(
            "Some mapping keys were not found as columns in the wide-format data: "
            f"{missing}. Available columns include: {sorted(list(available))[:20]} ..."
        )

    keep_cols = ["date"] + list(map_norm.keys())
    out_wide = df[keep_cols].copy()

    #melt to tidy
    out = out_wide.melt(id_vars=["date"], var_name="series_key", value_name="value")
    out["category"] = out["series_key"].map(map_norm)
    out["value"] = _coerce_numeric(out["value"])

    require_columns(out, ["date", "category", "value"], "ons_tidy_wide")
    require_non_null(out, ["date", "category"], "ons_tidy_wide")

    return out[["date", "category", "value"]]

#3. resample to weekly frequency
def to_weekly_demand(
        df_tidy: pd.DataFrame, 
        *, 
        date_col: str = "date",
        value_col: str = "value",
        agg: str = "mean",
        monthly_to_weekly_method: str = "ffill",
    ) -> pd.DataFrame:
    """
    Convert tidy ONS series to weekly demand time series with Monday week start.

    Supports:
     - if source weekly/daily, aggregate to weekly 
     - if source monthly, resample to weekly with justified method

    Parameters:
     - df_tidy : pd.DataFrame
        columns required: date, category, value
     - agg : str
        weekly agg: "mean" or "sum"
     - monthly_to_weekly_method : str
        method to resample monthly to weekly:
         - "ffill" : forward fill
         - "interpolate" : interpolate missing weeks
    
    Returns:
     - pd.DataFrame
        columns: 
         - week_start (datetime)
         - category (str)
         - demand (float) 
    """
    df = df_tidy.copy()
    require_columns(df, [date_col, "category", value_col], "ons_tidy")
    df[date_col] = _parse_date_series(df[date_col])
    df[value_col] = _coerce_numeric(df[value_col])

    require_non_null(df, [date_col, "category"], "ons_tidy")

    #monthly type: first of month OR frequncy month-end
    dates = df[date_col].dropna().sort_values()
    if dates.empty:
        raise ValueError("No valid dates in tidy demand data after parsing.")
    
    #simple monthly detection: majority on day 1
    day1_ratio = (dates.dt.day == 1).mean()
    is_monthly_like = day1_ratio > 0.8

    #convert to week start (Monday)
    df["week_start"] = _week_start_monday(df[date_col])

    #if monthly-like, resample to weekly
    if is_monthly_like:
        weekly_frames: list[pd.DataFrame] = []
        for category, group in df.groupby("category", sort=False):
            g2 = group[["week_start", value_col]].copy()
            g2 = g2.dropna(subset=["week_start"])
            g2 = g2.groupby("week_start", as_index=True)[value_col].mean().sort_index()

            #create full weekly index between min/max
            full_idx = pd.date_range(
                g2.index.min(),
                g2.index.max(),
                freq="W-MON"
            )
            g2 = g2.reindex(full_idx)

            if monthly_to_weekly_method == "ffill":
                g2 = g2.ffill()
            elif monthly_to_weekly_method == "interpolate":
                g2 = g2.interpolate(limit_direction="both")
            else:
                raise ValueError(
                    f"Unsupported monthly_to_weekly_method: {monthly_to_weekly_method}"
                    " Must be 'ffill' or 'interpolate'."
                )
            
            tmp = (
                g2.reset_index()
                .rename(columns={
                    "index": "week_start",
                    value_col: "demand"
                })
            )
            tmp["category"] = category
            weekly_frames.append(tmp)

        out = pd.concat(weekly_frames, ignore_index=True)

        require_columns(out, ["week_start", "category", "demand"], "demand_weekly_monthly_upsampled")
        require_non_null(out, ["week_start", "category"], "demand_weekly_monthly_upsampled")
        require_unique_keys(out, ["week_start", "category"], "demand_weekly_monthly_upsampled")

        return (
            out[["week_start", "category", "demand"]]
            .sort_values(["category", "week_start"])
            .reset_index(drop=True)
        )
    
    #otherwise: aggregate to weekly
    if agg not in {"mean", "sum"}:
        raise ValueError(f"Unsupported agg method: {agg}. Must be 'mean' or 'sum'.")
    
    if agg == "mean":
        grouped = df.groupby(["week_start", "category"], as_index=False)[value_col].mean()
    else:  #agg == "sum"
        grouped = df.groupby(["week_start", "category"], as_index=False)[value_col].sum()

    grouped = grouped.rename(columns={value_col: "demand"})

    require_unique_keys(grouped, ["week_start", "category"], "demand_weekly_aggregated")
    require_non_null(grouped, ["week_start", "category"], "demand_weekly_aggregated")

    return (
        grouped[["week_start", "category", "demand"]]
        .sort_values(["category", "week_start"])
        .reset_index(drop=True)
    )

#4. validate time series assumptions and save processed file
def build_demand_weekly(
    ons_path: Path,
    mapping: dict[str, str],
    out_path: Path,
    *,
    sheet_name: str | None = None,
    agg: str = "mean",
    monthly_to_weekly_method: str = "ffill",
    reader_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Orchestrator for demand ingestion:
      load raw -> map to categories -> convert to weekly -> validate -> save
    
    Parameters:
     - ons_path : Path
        Path to raw ONS retail data file
     - mapping : dict[str, str]
        Mapping from ONS series id to project category
     - out_path : Path
        Path to save processed parquet file
     - sheet_name : str | None
        optional sheet name for Excel files
     - agg : str
        weekly agg: "mean" or "sum"
    - monthly_to_weekly_method : str
        method to convert monthly data to weekly: "ffill" or "interpolate"
    - reader_kwargs : dict | None
        extra kwargs for pandas reader 

    Returns:
     - pd.DataFrame
        weekly tidy demand:
        - week_start (datetime)
        - category (str)
        - demand (float)
    """
    reader_kwargs = reader_kwargs or {}

    df_raw = load_ons_retail_raw(ons_path, sheet_name=sheet_name, **reader_kwargs)
    df_tidy = map_ons_to_categories(df_raw, mapping=mapping)
    df_weekly = to_weekly_demand(
        df_tidy,
        date_col="date",
        value_col="value",
        agg=agg,
        monthly_to_weekly_method=monthly_to_weekly_method,
    )

    #final checks
    require_columns(df_weekly, ["week_start", "category", "demand"], "demand_weekly_final")
    require_unique_keys(df_weekly, ["week_start", "category"], "demand_weekly_final")
    require_non_null(df_weekly, ["week_start", "category", "demand"], "demand_weekly_final")

    write_parquet(df_weekly, out_path, dataset_name="demand_weekly")
    return df_weekly
