#formalising files living under data/processed

from __future__ import annotations

import pandas as pd

def load_weekly_demand(
        path: str = "data/processed/demand_weekly.csv",
    ) -> pd.DataFrame:
    """Load weekly demand date
    Expects a csv with columns:
    - week_start
    - category
    - demand
    """
    df = pd.read_csv(path)

    if "week_start" not in df.columns or "category" not in df.columns or "demand" not in df.columns:
        raise ValueError(
            "Input data must contain columns:"
            " 'week_start', 'category', and 'demand'"
        )
    
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    return df