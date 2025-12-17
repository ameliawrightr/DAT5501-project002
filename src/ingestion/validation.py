import pandas as pd

#1. Ensure that the DataFrame contains the required columns.
def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

#2. Ensure that the combination of keys is unique in the DataFrame.
def require_unique_keys(df: pd.DataFrame, keys: list[str], name: str) -> None: 
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{name} contains duplicate entries for keys: {keys}")

#3. Ensure that there are no missing weeks in the time series data.
def require_no_missing_weeks(df: pd.DataFrame, week_col: str, name: str) -> None:
    all_weeks = pd.date_range(start=df[week_col].min(), end=df[week_col].max(), freq='W-MON')
    missing_weeks = set(all_weeks) - set(df[week_col])
    if missing_weeks:
        raise ValueError(f"{name} is missing weeks: {missing_weeks}")

#4. Ensure that specified columns do not contain null values.
def require_non_null(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for col in cols:
        if df[col].isnull().any():
            raise ValueError(f"{name} contains null values in column: {col}")

#5. Ensure that the values in a column fall within a specified range.
def require_numeric_range(df: pd.DataFrame, col: str, lo: float | None, hi: float | None, name: str) -> None:
    if lo is not None and (df[col] < lo).any():
        raise ValueError(f"{name} contains values in {col} less than {lo}")
    if hi is not None and (df[col] > hi).any():
        raise ValueError(f"{name} contains values in {col} greater than {hi}")