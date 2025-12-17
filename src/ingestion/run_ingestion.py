#ONE COMMAND TO RUN INGESTION END TO END

from pathlib import Path
import yaml

from .ons_retail import build_demand_weekly

def run_all(config_path: Path) -> None: 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    ons_cfg = config['ons_retail']

    build_demand_weekly(
        ons_path=Path(ons_cfg['file']),
        sheet_name=ons_cfg['sheet'],
        category_mapping=ons_cfg['mapping'],
        out_path=Path('data/processed/demand_weekly.parquet'),
    )
    

    