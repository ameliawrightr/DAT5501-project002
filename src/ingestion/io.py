#shared IO + validation functions
from __future__ import annotations

from pathlib import Path
import pandas as pd

#1. Ensure that a directory exists.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

#2. Read a CSV/Excel file into a DataFrame.
def read_csv_or_excel(
        path: str | Path,
        *,
        sheet_name: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
    """
    Read tabular file that may be CSV or Excel format.
     - if suffix is .csv -> use pd.read_csv
     - if suffix is .xlsx or .xls -> use pd.read_excel
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p, **kwargs)
    
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet_name, **kwargs)
    
    #fallback: try to infer from sheet_name presence
    raise ValueError(
        f"Unsupported file type for {p.name!r} (suffix: {suffix!r})."
        "Expected .csv, .xlsx, or .xls."
    )


#3. Write a DataFrame to a CSV file.
def write_csv(
        df: pd.DataFrame,
        out_path: str | Path,
        *,
        dataset_name: str | None = None
    ) -> None:
    #write df as csv, ensuring parent dirs exists
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p, index=False)

    if dataset_name:
        print(f"[write_csv] Wrote {dataset_name!r} CSV file to {p.resolve()}")
    else:
        print(f"[write_csv] Wrote CSV file to {p.resolve()}")

    print(f"[write_csv] Wrote CSV file to {p.resolve()}")


