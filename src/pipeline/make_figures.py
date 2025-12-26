# Generate the 4 key figures
#1. Overall error by model (sMAPE)
#2. Event vs non-event error (sMAPE)
#3. Stability plot: per-origin error for baseline vs event_ridge
#4. One “forecast vs actual” plot during an event window for each category

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
    #overall error bar charts by model, using sMAPE (%)
    #only plot key models to avoid clutter

    metric = "sMAPE"
    pretty_metric = "Mean sMAPE (%)"

    key_models = [
        "seasonal_naive",
        "rolling_average",
        "event_ridge",
        "event_random_forest",
    ]

    for cat, g in overall.groupby("category"):
        g = g[g["model"].isin(key_models)].copy()
        if g.empty:
            continue 

        label_map = {
            "seasonal_naive": "Seasonal Naive",
            "rolling_average": "Rolling Average",
            "event_ridge": "Event Ridge",
            "event_random_forest": "Random Forest (event-aware)",
        }
        g["model_label"] = g["model"].map(label_map).fillna(g["model"])

        g = g.sort_values(metric, ascending=True)

        x = np.arange(len(g), dtype=float)
        values = g[metric].values
        
        plt.figure(figsize=(8, 4.5))
        plt.bar(x, values)
        plt.ylabel(pretty_metric)
        plt.title(f"Overall {pretty_metric} by model — {cat}")
        plt.xticks(x, g["model_label"], rotation=20, ha="right")

        #add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(
                x[i],
                v + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.tight_layout()
        out = FIG_DIR / f"{cat}_overall_sMAPE_by_model.png"
        plt.savefig(out, dpi=200)
        plt.close()

#2. Event vs non-event (baseline vs event_ridge)
def fig_event_vs_nonevent_mae(ev: pd.DataFrame) -> None:
    #event vs non event error comparison (sMAPE) for two main models:
    # baseline Ridge vs event-aware Random Forest
    metric = "sMAPE"
    pretty_metric = "Mean sMAPE (%)"

    models_of_interest = ["event_ridge", "event_random_forest"]
    subset_order = ["event_weeks", "non_event_weeks"]

    for cat, gcat in ev.groupby("category"):
        gcat = gcat[gcat["model"].isin(models_of_interest)].copy()
        if gcat.empty:
            continue

        #build arrays: [baseline, RF] x [event, non-event]
        models_labels = ["Ridge (baseline)", "Random Forest (event-aware)"]
        x = np.arange(len(models_of_interest), dtype=float)
        width = 0.35

        vals_event = []
        vals_nonevent = []

        for m in models_of_interest:
            gm = gcat[gcat["model"] == m]
            me = gm.loc[gm["subset"] == "event_weeks", metric].values
            mn = gm.loc[gm["subset"] == "non_event_weeks", metric].values
            vals_event.append(float(me[0]) if len(me) else np.nan)
            vals_nonevent.append(float(mn[0]) if len(mn) else np.nan)

        
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - width / 2, vals_event, width, label="Event weeks")
        plt.bar(x + width / 2, vals_nonevent, width, label="Non-event weeks")
        
        plt.ylabel(pretty_metric)
        plt.title(f"{pretty_metric}: event vs non-event — {cat}")
        plt.xticks(x, models_labels, rotation=15, ha="right")
        plt.legend()

        #add value labels on top of bars
        for i, v in enumerate(vals_event):
            if np.isnan(v):
                continue
            plt.text(x[i] - width / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        for i, v in enumerate(vals_nonevent):
            if np.isnan(v):
                continue
            plt.text(x[i] + width / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out = FIG_DIR / f"{cat}_event_vs_nonevent_sMAPE.png"
        plt.savefig(out, dpi=200)
        plt.close()

#3. Stability plot
def fig_origin_stability(stab: pd.DataFrame) -> None:
    #stability mfigure using summary per origin statistics
    # show mean MAE per origin with errors bars (std) for key models
    metric = "origin_MAE_mean"
    err_metric = "origin_MAE_std"
    pretty_metric = "Mean MAE per origin"

    key_models = [
        "seasonal_naive",
        "rolling_average",
        "event_ridge",
        "event_random_forest",
    ]

    label_map = {
        "seasonal_naive": "Seasonal Naive",
        "rolling_average": "Rolling Average",
        "event_ridge": "Event Ridge",
        "event_random_forest": "Random Forest (event-aware)",
    }

    for cat, g in stab.groupby("category"):
        g = g[g["model"].isin(key_models)].copy()
        if g.empty:
            continue 

        g["model_label"] = g["model"].map(label_map).fillna(g["model"])
        g = g.sort_values(metric, ascending=True)

        x = np.arange(len(g), dtype=float)
        values = g[metric].values
        errors = g[err_metric].values

        plt.figure(figsize=(8, 5))
        plt.bar(x, values, yerr=errors, capsize=3)
        plt.ylabel(pretty_metric)
        plt.title(f"Forecast stability across origins — {cat}")
        plt.xticks(x, g["model_label"], rotation=20, ha="right")

        #label bars with mean MAE values
        for i, v in enumerate(values):
            plt.text(
                x[i],
                v + errors[i] + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

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

    stability = _load_summary("stability_metrics.csv")
    fig_origin_stability(stability)

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
