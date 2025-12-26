# Generate the 4 key figures
#1. Bar chart: mean MAE by model for each category
#2. Event vs non-event MAE (baseline vs event_ridge)
#3. One “forecast vs actual” plot during an event window for each category
#4. Stability plot: per-origin MAE for baseline vs event_ridge

from __future__ import annotations

from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BACKTEST_DIR = Path("artifacts/backtests")
SUMMARY_DIR = Path("artifacts/summary")
FIG_DIR = Path("artifacts/figures")


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_summary(name: str) -> pd.DataFrame:
    path = SUMMARY_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run summarise_backtests first.")
    return pd.read_csv(path)


def _load_detailed_for(category: str, model: str) -> pd.DataFrame:
    path = BACKTEST_DIR / f"{category}_{model}_detailed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing detailed CSV: {path}")
    df = pd.read_csv(path)
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    return df


def _event_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("is_")]

#1. Bar chart: mean MAE by model for each category
def fig_overall_mae_bar(overall: pd.DataFrame) -> None:
    #one figure per category, bar chart of MAE by model (sorted)
    for cat, g in overall.groupby("category"):
        g = g.sort_values("MAE", ascending=True).copy()
        plt.figure(figsize=(8, 4.5))
        plt.bar(g["model"], g["MAE"])
        plt.ylabel("Mean MAE")
        plt.title(f"Overall MAE by model — {cat}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = FIG_DIR / f"{cat}_overall_mae_by_model.png"
        plt.savefig(out, dpi=200)
        plt.close()

#2. Event vs non-event MAE (baseline vs event_ridge)
def fig_event_vs_nonevent_mae(ev: pd.DataFrame) -> None:
    #for each category, compare MAE for event vs non-event weeks, grouped by model
    for cat, gcat in ev.groupby("category"):
        models = list(gcat["model"].unique())
        models.sort()

        subset_order = ["event_weeks", "non_event_weeks"]
        # positions
        x = np.arange(len(models), dtype=float)
        width = 0.38

        # build aligned arrays
        vals_event = []
        vals_nonev = []
        for m in models:
            gm = gcat[gcat["model"] == m]
            me = gm.loc[gm["subset"] == "event_weeks", "MAE"].values
            mn = gm.loc[gm["subset"] == "non_event_weeks", "MAE"].values
            vals_event.append(float(me[0]) if len(me) else np.nan)
            vals_nonev.append(float(mn[0]) if len(mn) else np.nan)

        plt.figure(figsize=(9, 4.8))
        plt.bar(x - width/2, vals_event, width, label="Event weeks")
        plt.bar(x + width/2, vals_nonev, width, label="Non-event weeks")
        plt.ylabel("Mean MAE")
        plt.title(f"Event vs non-event MAE — {cat}")
        plt.xticks(x, models, rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        out = FIG_DIR / f"{cat}_event_vs_nonevent_mae.png"
        plt.savefig(out, dpi=200)
        plt.close()

#3. Stability plot: per-origin MAE for baseline vs event_ridge
def fig_origin_stability(detailed_paths: list[Path], highlight_models: list[str] | None = None) -> None:
    #plot per-origin MAE lines (stability) for each category.
    #uses detailed CSVs: compute MAE per origin_time.

    if highlight_models is None:
        highlight_models = ["seasonal_naive", "rolling_average", "event_ridge"]

    #find categories by scanning filenames
    # - expects: <category>_<model>_detailed.csv
    triples = []
    for p in detailed_paths:
        name = p.name.replace("_detailed.csv", "")
        parts = name.split("_")
        if len(parts) < 2:
            continue
        cat = parts[0]
        model = "_".join(parts[1:])
        triples.append((cat, model, p))

    df_meta = pd.DataFrame(triples, columns=["category", "model", "path"])
    for cat, gcat in df_meta.groupby("category"):
        #only selected models if present
        gsel = gcat[gcat["model"].isin(highlight_models)]
        if gsel.empty:
            continue

        plt.figure(figsize=(9, 4.8))
        for _, row in gsel.iterrows():
            df = pd.read_csv(row["path"])
            df["origin_time"] = pd.to_datetime(df["origin_time"])
            # per-origin MAE
            per_origin = (
                df.groupby("origin_time")[["y_true", "y_pred"]]
                .apply(lambda x: np.mean(np.abs(x["y_true"].astype(float) - x["y_pred"].astype(float))))
                .reset_index(name="origin_MAE")
                .sort_values("origin_time")
            )
            plt.plot(per_origin["origin_time"], per_origin["origin_MAE"], label=row["model"])

        plt.ylabel("MAE per origin")
        plt.title(f"Stability across backtest origins — {cat}")
        plt.legend()
        plt.tight_layout()
        out = FIG_DIR / f"{cat}_origin_stability_mae.png"
        plt.savefig(out, dpi=200)
        plt.close()

#4. One “forecast vs actual” plot during an event window for each category
def fig_example_forecast_paths(category: str, model: str, n_origins: int = 6) -> None:
    #show multiple forecast paths (y_pred) vs actuals for a small number of origins.
    #this produces a clean visual of backtest behaviour without needing a separate 'actual series' file.

    df = _load_detailed_for(category, model)
    df = df.sort_values(["origin_time", "forecast_time"])

    #take last N origins for a modern-looking plot
    origins = sorted(df["origin_time"].unique())[-n_origins:]
    df = df[df["origin_time"].isin(origins)]

    #build an "actuals" line for the forecast_time points in this slice
    actual = (
        df.groupby("forecast_time")["y_true"]
        .mean()  
        .sort_index()
    )

    plt.figure(figsize=(10, 5.2))
    plt.plot(actual.index, actual.values, label="Actual")

    for origin in origins:
        g = df[df["origin_time"] == origin].sort_values("forecast_time")
        plt.plot(g["forecast_time"], g["y_pred"].astype(float).values, label=f"Origin {origin.date()}")

    plt.title(f"Forecast paths vs actual — {category} ({model})")
    plt.ylabel("Demand")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = FIG_DIR / f"{category}_{model}_forecast_paths.png"
    plt.savefig(out, dpi=200)
    plt.close()

#4b. One “forecast vs actual” plot during an event window for each category
def fig_event_window_example(category: str, model: str) -> None:
    #Plot forecast errors around an event flag for interpretability.
    #Picks the first is_* column found and plots abs_error over time with event markers.
    df = _load_detailed_for(category, model)
    evcols = _event_cols(df)
    if not evcols:
        return

    #choose the most "active" event column in this detailed file
    counts = {c: int(df[c].astype(bool).sum()) for c in evcols}
    event_col = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]

    #aggregate abs_error by forecast_time and whether that week is an event
    df[event_col] = df[event_col].astype(bool)
    agg = (
        df.groupby("forecast_time")[["abs_error", event_col]]
        .agg({"abs_error": "mean", event_col: "max"})
        .reset_index()
        .sort_values("forecast_time")
    )

    plt.figure(figsize=(10, 4.8))
    plt.plot(agg["forecast_time"], agg["abs_error"], label="Mean abs error")
    #overlay event points
    ev = agg[agg[event_col]]
    if not ev.empty:
        plt.scatter(ev["forecast_time"], ev["abs_error"], label=f"Event weeks ({event_col})", marker="o")

    plt.title(f"Error over time with event weeks highlighted — {category} ({model})")
    plt.ylabel("Mean abs error")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"{category}_{model}_error_with_events.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    _ensure_dirs()

    overall = _load_summary("overall_metrics.csv")
    ev = _load_summary("event_vs_nonevent_metrics.csv")

    fig_overall_mae_bar(overall)
    fig_event_vs_nonevent_mae(ev)

    detailed_paths = [Path(p) for p in glob.glob(str(BACKTEST_DIR / "*_detailed.csv"))]
    fig_origin_stability(detailed_paths)

    #example forecast paths and event window plots
    examples = [
        ("fitness_equipment", "event_ridge"),
        ("electronic_goods", "event_ridge"),
        ("school_supplies", "event_ridge"),
    ]
    for cat, model in examples:
        try:
            fig_example_forecast_paths(cat, model, n_origins=6)
            fig_event_window_example(cat, model)
        except FileNotFoundError:
            # skip if filenames differ
            continue

    print(f"[OK] Figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
