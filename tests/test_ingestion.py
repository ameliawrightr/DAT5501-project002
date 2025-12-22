import pandas as pd

from src.ingestion.demand import load_weekly_demand

#Load weekly demand dataset and validate it has required columns
def load_and_validate(path: str) -> pd.DataFrame:
    df = load_weekly_demand("data/processed/demand_monthly.csv")
    
    #basic expectations:
    assert not df.empty
    assert {"week_start", "category", "demand"}.issubset(df.columns)

    #week_start should be datetime
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])

    #no negative demand
    assert (df["demand"] >= 0).all()

