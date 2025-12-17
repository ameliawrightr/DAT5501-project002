#ONE COMMAND TO RUN INGESTION END TO END
"""
RUNS:
- build_demand_weekly
- build_events_weekly
"""

from pathlib import Path

def run_all(config_path: Path) -> None: ...
    