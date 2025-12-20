#ONE COMMAND TO RUN INGESTION END TO END

from pathlib import Path
import yaml

from .ons_retail import build_demand_weekly


def load_config(config_path: Path | None = None) -> dict:
    #project root = parent of src/
    project_root = Path(__file__).resolve().parents[2]

    if config_path is None:
        #default: project_root/config.yaml
        config_path = project_root / 'config.yaml'
    else:
        #allow relative paths - interpret relative to project root
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
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
    ons_cfg = config['ons_retail']

    build_demand_weekly(
        ons_path=Path(ons_cfg['file']),
        sheet_name=ons_cfg.get('sheet'),
        mapping=ons_cfg['mapping'],
        out_path=Path('data/processed/demand_weekly.parquet'),
    )
    
if __name__ == "__main__":
    #uses default project_root/config.yaml
    run_all()
    