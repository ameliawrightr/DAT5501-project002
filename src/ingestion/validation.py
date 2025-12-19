from __future__ import annotations
import pandas as pd

class ValidationError(ValueError):
    #Raised when df fails pipeline validation checks
    pass

#1. Ensure that the DataFrame contains the required columns.
def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValidationError(
            f"{name} is missing required columns: {missing}."
            f"Available columns: {list(df.columns)}"
            )

#2. Ensure that the combination of keys is unique in the DataFrame.
def require_unique_keys(df: pd.DataFrame, keys: list[str], name: str) -> None: 
    require_columns(df, keys, name)
    
    #drop rows where any key is null-null, keys are always invalid for uniqueness checks
    key_df = df[keys]
    if key_df.isna().any(axis=None):
        null_rows = int(key_df.isna().any(axis=1).sum())
        raise ValidationError(f"{name} contains null values in keys: {keys} ({null_rows} rows)")

    dup_mask = df.duplicated(subset=keys, keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, keys].head(10).to_dict(orient='records')
        raise ValidationError(
            f"[{name}] contains duplicate entries for keys: {keys}."
            f"Example duplicates (first 10): {dup_rows} ")

#3. Ensure that there are no missing weeks in the time series data.
def require_no_missing_weeks(df: pd.DataFrame, week_col: str, name: str) -> None:
    #ensure weekly continuity in week_col
    require_columns(df, [week_col], name)

    s = pd.to_datetime(df[week_col], errors='coerce')

    if s.isna().any():
        bad = int(s.isna().sum())
        raise ValidationError(f"[{name}] {bad} contains invalid dates in '{week_col}'.")
    
    if s.empty:
        raise ValidationError(f"[{name}] '{week_col}' is empty; cannot validate weekly continuity.")
    
    min_w = s.min()
    max_w = s.max()

    expected = pd.date_range(start=min_w, end=max_w, freq='W-MON')
    observed = pd.Index(sorted(s.unique()))

    missing = expected.difference(observed)
    if len(missing) > 0:
        sample = [d.strftime('%Y-%m-%d') for d in missing[:10]]
        raise ValidationError(
            f"[{name}] Missing {len(missing)} week(s) in '{week_col}' "
            f"between {min_w.date()} and {max_w.date()}. "
            f"Examples: {sample}"
        )

#4. Ensure that specified columns do not contain null values.
def require_non_null(df: pd.DataFrame, cols: list[str], name: str) -> None:
    require_columns(df, cols, name)
    null_counts = df[cols].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        details = ", ".join([f"{col} ({count} nulls)" for col, count in bad.items()])
        raise ValidationError(f"{name} contains null values in columns: {details}")

#5. Ensure that the values in a column fall within a specified range.
def require_numeric_range(
        df: pd.DataFrame, 
        col: str, 
        lo: float | None, 
        hi: float | None, 
        name: str
) -> None:
    #ensure column is numeric and within optional [lo, hi] range
    require_columns(df, [col], name)
    
    s = pd.to_numeric(df[col], errors='coerce')
    if s.isna().all():
        bad = int(s.isna().sum())
        raise ValidationError(f"[{name}] column '{col}' cannot be converted to numeric;"
                              f"Contains {bad} invalid values (non-numeric or NaN).")
    
    if lo is not None:
        below = int((s < lo).sum())
        if below > 0:
            mn = float(s.min())
            raise ValidationError(
                f"[{name}] column '{col}' has {below} values below {lo}."
                f"Minimum value found: {mn}."
            )
        
    if hi is not None:
        above = int((s > hi).sum())
        if above > 0:
            mx = float(s.max())
            raise ValidationError(
                f"[{name}] column '{col}' has {above} values above {hi}."
                f"Maximum value found: {mx}."
            )
        

