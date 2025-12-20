#ONE COMMAND TO RUN INGESTION END TO END

from __future__ import annotations
from pathlib import Path

from .ons_retail import build_demand_monthly
from .calendar_events import build_events_monthly

from .validation import (
    require_columns,
    require_non_null,
    require_unique_keys,
)

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    demand_in = project_root / "data" / "processed" / "retail_volume_monthly_tidy.csv"
    demand_out = project_root / "data" / "processed" / "demand_monthly.csv"

    events_in = project_root / "data" / "processed" / "calendar_events_uk_weekly_1988_2025.csv"
    events_out = project_root / "data" / "processed" / "events_monthly.csv"

    if not demand_in.exists():
        raise FileNotFoundError(f"Demand input file not found: {demand_in}")
    if not events_in.exists():
        raise FileNotFoundError(f"Events input file not found: {events_in}")
    
    build_demand_monthly(demand_in, demand_out)
    build_events_monthly(events_in, events_out)

if __name__ == "__main__":
    main()