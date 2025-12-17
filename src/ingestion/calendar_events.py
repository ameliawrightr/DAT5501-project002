#CALENDAR + EVENTS INDICATOR INGESTION

from pathlib import Path
import pandas as pd

#1. bank holidays
def load_bank_holidays(source: str | Path) -> pd.DataFrame:
    """
    load raw bank holidays file from source
    returns: date, title, region
    """
    pass

#2. create weekly holiday indicator
def bank_holidays_to_weekly(df_holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: week_start, is_bank_holiday_week (0/1)
    """

#3. encode manual events windows
def build_event_windows(
        start: str,
        end: str,
        *,
        new_year_weeks: tuple[int, int] = (52, 1),
        back_to_school_months: tuple[int, int] = (8, 9),
        exam_months: tuple[int, int] = (5, 6),
        q4_months: tuple[int, int] = (10, 11, 12),  
    ) -> pd.DataFrame:
    """
    returns: 
      - week_start + 4 binary event columns
      - definition = explicit + reproducible
    """
    pass

#4. merge into single calendar events table and save
def build_events_weekly(
    bank_holiday_source: str | Path,
    out_path: Path,
    *,
    start: str,
    end: str,  
) -> pd.DataFrame:
    """
    holidays -> weekly
    event windows -> weekly
    merge -> validate -> write -> return
    """