#ONE COMMAND TO RUN INGESTION END TO END

from pathlib import Path
import yaml

from .ons_retail import build_demand_weekly

def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
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
    config_path = Path("/Users/amelia/DAT5501-project002/.circleci/config.yaml")
    run_all(config_path)

    