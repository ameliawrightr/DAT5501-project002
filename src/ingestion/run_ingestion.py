"""
ONE COMMAND TO RUN INGESTION END TO END
This script builds the core processed datasets used by the project:

1) demand_monthly.csv
   - standardised monthly demand (ONS retail volume extract)
   - optionally enriched with monthly event features (depending on your pipeline)

2) events_monthly.csv
   - monthly event indicators aggregated from the weekly event calendar

Run from the project root (recommended):
    python -m src.ingestion.run_ingestion

Or directly:
    python src/ingestion/run_ingestion.py
"""

from __future__ import annotations
from pathlib import Path

from .ons_retail import build_demand_monthly
from .calendar_events import build_events_monthly

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