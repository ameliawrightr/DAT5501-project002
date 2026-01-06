# Event-Aware Demand Forecasting — Coursework Repo

This repo contains a small, reproducible forecasting pipeline for event-driven retail demand:
- **Ingestion** builds standardised datasets under `data/processed/`
- **Models** include baselines + event-aware models (calendar flags)
- **Evaluation** uses rolling-origin backtesting and generates **artifacts** (CSVs + figures)
- **CI** (CircleCI) runs tests on every commit

## Structure (high level)
- `src/` — ingestion, features, models, evaluation
- `data/processed/` — processed inputs used by the pipeline (CSV)
- `artifacts/` — generated outputs (backtests, summaries, figures)
- `.circleci/` — CI pipeline config
- `tests/` — unit/integration tests
- `requirements.txt` — dependencies

## How to run
Create env + install:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Run ingestion (writes to `data/processed/`):
```bash
python -m src.ingestion.run_ingestion
```

Run rolling-origin backtests (writes to `artifacts/backtests/`):
```bash
python -m src.models.run_backtests
```

Summarise results (writes to `artifacts/summary/`) *(if included in repo)*:
```bash
python -m src.evaluation.summarise_backtests
```

Generate figures (writes to `artifacts/figures/`):
```bash
python -m src.evaluation.make_figures
```

Run tests (same as CI):
```bash
pytest -q
```

## Outputs (artifacts)
Generated files are written under `artifacts/`:
- `artifacts/backtests/` — detailed + aggregate rolling-origin CSVs per model/category
- `artifacts/summary/` — aggregated metrics used for plotting (overall / event vs non-event / stability)
- `artifacts/figures/` — PNG figures used in the report

> Tip: For coursework, it’s common to **commit figures** but ignore large backtest CSVs.
