#formalising files living under data/processed

from __future__ import annotations

import pandas as pd

def load_weekly_demand(
        demand_csv_path: str = "data/processed/demand_monthly.csv",
    ) -> pd.DataFrame:
    """Load weekly demand data

    Supports two common layouts:
    1. Long format with columns: week_start, category, demand
    2. Wide format with week_start, fitness_equipment, electronics_accessories, school_supplies

    Parameters:
    - path: str
        Path to the CSV file
    Returns:
    - pd.DataFrame
        Columns: week_start, category, demand
    """
    df = pd.read_csv(demand_csv_path)

    #1) find date column
    date_col = None
    for cand in ["week_start", "week", "week_commencing", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    
    if date_col is None:
        raise ValueError(
            "No date column found in data."
            "Expected one of: week_start, week, week_commencing, date"
            f"Got: {df.columns.tolist()}"
        )
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "week_start"})

    #2. if already long - tidy and return
    if {"week_start", "category", "demand"}.issubset(df.columns):
        out = df[["week_start", "category", "demand"]].copy()
        out = out.sort_values("week_start")
        return out
    
    #3. else assume wide format: first col = date, other = categories
    non_date_cols = [col for col in df.columns if col != "week_start"]

    if len(non_date_cols) == 0:
        raise ValueError(
            "No demand columns found.",
            "Expected either 'category'+'demand' or category columns per product."
        )
    
    long_df = df.melt(
        id_vars=["week_start"],
        value_vars=non_date_cols,
        var_name="category",
        value_name="demand",    
    ).dropna(subset=["demand"])

    long_df = long_df.sort_values("week_start")

    return long_df
   
   