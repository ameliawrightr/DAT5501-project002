#ONE COMMAND TO RUN INGESTION END TO END

from pathlib import Path
import yaml

from .ons_retail import build_demand_weekly
from .calendar_events import build_events_weekly


def load_config(config_path: Path | None = None) -> dict:
    #project root = parent of src/
    project_root = Path(__file__).resolve().parents[2]
    config_path = (project_root / 'config.yaml') if config_path is None else config_path

    print(f"[run_ingestion] Loading config from {config_path}") #debug print
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file {config_path} is empty or not a YAML mapping."
            f"Got: {config!r}"
         )
    return config


def run_all(config_path: Path | None = None) -> None: 
    config = load_config(config_path)

    d = config["demand"]
    e = config["events"]

    #build demand weekly
    build_demand_weekly(
        tidy_path=Path(d["in_file"]),
        out_path=Path(d["out_file"]),
        date_col=d.get("date_col", "date"),
        category_col=d.get("category_col", "category"),
        value_col=d.get("value_col", "value"),
        monthly_to_weekly_method=d.get("monthly_to_weekly_method", "ffill"),
        category_remap=d.get("category_remap"),
    )

    #build events weekly
    build_events_weekly(
        in_path=Path(e["in_file"]),
        out_path=Path(e["out_file"]),
        week_start_col=e.get("week_start_col", "week_start"),
        dayfirst=bool(e.get("dayfirst", True)),
    )

    
if __name__ == "__main__":
    #uses default project_root/config.yaml
    run_all()
    