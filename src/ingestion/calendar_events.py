#CALENDAR + EVENTS INDICATOR INGESTION
from __future__ import annotations
from calendar import month
from pathlib import Path
import pandas as pd

from .validation import (
    ValidationError,
    require_columns,
    require_unique_keys,
    require_no_missing_weeks,
    require_non_null,
    require_numeric_range,  
)
from .io import write_parquet


#1. construct weekly event indicators for event driven demand forecasting
def build_event_windows(
        start: str,
        end: str,
        *,
        #New Year: ISO weeks 52-1
        new_year_weeks: tuple[int, int] = (52, 1),
        #back to school: Aug-Sept
        back_to_school_months: tuple[int, int] = (8, 9),
        #exam period: May-June
        exam_months: tuple[int, int] = (5, 6),
        #Q4: Nov-Dec
        q4_months: tuple[int, int] = (10, 11, 12),
    ) -> pd.DataFrame:
    """
    Def:
        - New Year: ISO weeks 1-3 (fitness equipment)
        - Back to school: Aug-Sept (school supplies, electronics)
        - Exam period: May-June (school supplies)
        - Q4: Nov-Dec (electronics)
    """
    weeks = pd.date_range(
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
        freq="W-MON"
    )

    #ISO year/week + month
    iso = weeks.isocalendar()
    df = pd.DataFrame(
        {
            "week_start": weeks,
            "iso_year": iso.year,
            "iso_week": iso.week,
            "month": weeks.month,
        }
    )

    #event flags
    df["is_new_year"] = (
        (df["iso_week"] >= new_year_weeks[0]) &
        (df["iso_week"] <= new_year_weeks[1])
    ).astype(int)

    df["is_back_to_school"] = df["month"].isin(back_to_school_months).astype(int)
    df["is_exam_period"] = df["month"].isin(exam_months).astype(int)
    df["is_q4"] = df["month"].isin(q4_months).astype(int)

    require_unique_keys(df, ["week_start"], "event_windows")

    return df

#2. merge into single calendar events table and save
def build_events_weekly(
    out_path: Path,
    *,
    start: str,
    end: str,  
) -> pd.DataFrame:
    """
    orchestrate construction and persistence of weekly event indicators
    """
    df = build_event_windows(
        start=start,
        end=end
    )
    write_parquet(
        df,
        out_path,
        dataset_name="events_weekly"
    )
    return df


def main() -> None:
    out_path = Path("data/processed/events_weekly.parquet")

    start = "1988-01-01"
    end = "2025-10-31"

    build_events_weekly(
        out_path=out_path,
        start=start,
        end=end
    )

if __name__ == "__main__":
    main()