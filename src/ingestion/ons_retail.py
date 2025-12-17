#DEMAND INGESTION - ONS Retail Sales Data

from pathlib import Path
import pandas as pd

#1. load raw ONS file
def load_ons_retail_raw(pat: Path, *, sheet: str | None = None) -> pd.DataFrame:
    """
    Read raw ONS retail data file into dataframe.
    No reshaping - just loading.
    """
    pass

#2. map ONS series to 3 categories
def map_ons_to_categories(df_raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    mapping: dict of {ons_series_name_or_code: category_slug}
    Returns a tidy dataframe with columns:
      - date (timestamp as provided by ONS)
      - category
      - value
    """
    pass

#3. resample to weekly frequency
def to_weekly_demand(df_tidy: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """
    Produces weekly demand with week_start (Monday), category, demand.
    If source is monthly, resample to weekly with a justified method (e.g., forward fill),
    and record that choice in docs.
    """
    pass

#4. validate time series assumptions and save processed file
def build_demand_weekly(
    ons_path: Path,
    mapping: dict[str, str],
    out_path: Path,
    *,
    sheet: str | None = None,
) -> pd.DataFrame:
    """
    Orchestrator: load -> map -> weekly -> validate -> write -> return
    """
    pass


