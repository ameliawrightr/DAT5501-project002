# def load_and_validate --> 
"""
 - checks long format assumptions
 - standardises category names
 - sort and index time series
 - prepare for resampling
"""

import pandas as pd

def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"date", "category", "volume_index"}
    assert required_cols.issubset(df.columns)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["category", "date"])
    
    assert not df.duplicated(["date", "category"]).any()
    
    return df

