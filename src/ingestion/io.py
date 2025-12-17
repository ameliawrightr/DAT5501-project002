#1. shared IO + validation functions

from pathlib import Path
import pandas as pd

#1. Ensure that a directory exists.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

#2. Read a CSV file into a DataFrame.
def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

#3. Write a DataFrame to a Parquet file.
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


