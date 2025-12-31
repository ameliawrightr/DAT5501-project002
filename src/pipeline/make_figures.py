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

MODEL_ORDER = [
    "seasonal_naive",
    "rolling_average",
    "time_regression",
    "event_ridge",
    "event_random_forest",
]

MODEL_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "rolling_average": "Rolling Average",
    "time_regression": "Time Regression",
    "event_ridge": "Event Ridge",
    "event_random_forest": "Random Forest \n(event-aware)",
}

CATEGORY_LABELS = {
    "fitness_equipment": "Fitness Equipment",
    "fitness": "Fitness Equipment",
    "electronic_goods": "Electronic Accessories",
    "electronics": "Electronic Accessories",
    "electronic": "Electronic Accessories",
    "school_supplies": "School Supplies",
    "school": "School Supplies",
}

def _pretty_cat(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, CATEGORY_LABELS.get(cat, cat))

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

    #plot all core models (baseline + event-aware)
    key_models = MODEL_ORDER

    for cat, g in overall.groupby("category"):
        g = g[g["model"].isin(key_models)].copy()
        if g.empty:
            continue 

        #consistent order (by performance) 
        g = g.sort_values(metric, ascending=True)
        g["model_label"] = g["model"].map(MODEL_LABELS).fillna(g["model"])

        x = np.arange(len(g), dtype=float)
        values = g[metric].astype(float).values
        
        plt.figure(figsize=(10, 5.0))
        plt.bar(x, values)
        plt.ylabel(pretty_metric)
        plt.title(f"Overall {pretty_metric} by model — {_pretty_cat(cat)}")
        plt.xticks(x, g["model_label"], ha="center")
        
        #extend y-axis dynamically to fit labels
        y_max = float(np.nanmax(values)) if len(values) else 1.0
        plt.ylim(0, y_max * 1.25)

        #add value labels on top of bars
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            plt.text(
                x[i],
                v + 0.03 * y_max,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        plt.tight_layout()
        out = FIG_DIR / f"{cat}_overall_sMAPE_by_model.png"
        plt.savefig(out, dpi=300)
        plt.close()

#2. Event vs non-event (baseline vs event_ridge)
def fig_event_vs_nonevent_mae(ev: pd.DataFrame) -> None:
    #event vs non event error comparison (sMAPE) for two main models:
    # baseline Ridge vs event-aware Random Forest
    metric = "sMAPE"
    pretty_metric = "Mean sMAPE (%)"

    #plot only relevant models in consisent order
    ev = ev[ev["model"].isin(MODEL_ORDER)].copy()
    if ev.empty:
        return
    
    for cat, gcat in ev.groupby("category"):
        #pivot to: rows=models, cols=subset (event_weeks/non_event_weeks)
        wide = (
            gcat.pivot(index="model", columns="subset", values=metric)
            .reindex(MODEL_ORDER)
        )

        #some combos may be missing; drop models without both values
        wide = wide.dropna(how="all")
        if wide.empty:
            continue

        x = np.arange(len(wide.index), dtype=float)
        width = 0.35

        vals_event = wide.get("event_weeks", pd.Series(index=wide.index, dtype=float)).values
        vals_nonevent = wide.get("non_event_weeks", pd.Series(index=wide.index, dtype=float)).values

        plt.figure(figsize=(12, 5.0))
        plt.bar(x - width / 2, vals_event, width, label="Event weeks")
        plt.bar(x + width / 2, vals_nonevent, width, label="Non-event weeks")
        
        plt.ylabel(pretty_metric)
        plt.title(f"{pretty_metric}: event vs non-event — {_pretty_cat(cat)}")
        plt.xticks(x, [MODEL_LABELS.get(m, m) for m in wide.index], ha="center")
        plt.legend()

        # dynamic y-lim
        all_vals = np.r_[vals_event, vals_nonevent]
        all_vals = all_vals[~np.isnan(all_vals)]
        
        y_max = float(all_vals.max()) if all_vals.size else 1.0
        top = y_max * 1.25
        plt.ylim(0, top)

        #add value labels on top of bars
        for i, v in enumerate(vals_event):
            if np.isnan(v):
                continue
            plt.text(
                x[i] - width / 2, v + 0.02 * y_max, f"{v:.1f}", ha="center", va="bottom", fontsize=8
            )

        for i, v in enumerate(vals_nonevent):
            if np.isnan(v):
                continue
            plt.text(
                x[i] + width / 2, v + 0.02 * y_max, f"{v:.1f}", ha="center", va="bottom", fontsize=8
            )

        #annotate delta % above pair
        for i, (ve, vn) in enumerate(zip(vals_event, vals_nonevent)):
            if np.isnan(ve) or np.isnan(vn) or vn == 0:
                continue
            delta = (ve - vn) / vn * 100.0
            plt.text(
                x[i], max(ve, vn) + 0.06 * y_max, f"Δ {delta:.0f}%", 
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="red" if delta > 0 else "green"
            )

        plt.tight_layout()
        out = FIG_DIR / f"{cat}_event_vs_nonevent_sMAPE.png"
        plt.savefig(out, dpi=300)
        plt.close()

#3. Stability plot
#origin level MAE mean +- std (lower is better)
def fig_origin_stability(stab: pd.DataFrame) -> None:
    metric = "origin_MAE_mean"
    err_metric = "origin_MAE_std"
    pretty_metric = "Origin-level MAE (mean ± std)"

    key_models = MODEL_ORDER

    for cat, g in stab.groupby("category"):
        g = g[g["model"].isin(key_models)].copy()
        if g.empty:
            continue 

        g = g.sort_values(metric, ascending=True)
        g["model_label"] = g["model"].map(MODEL_LABELS).fillna(g["model"])

        x = np.arange(len(g), dtype=float)
        values = g[metric].astype(float).values
        errors = g[err_metric].astype(float).values

        plt.figure(figsize=(10, 5.0))
        plt.bar(x, values, yerr=errors, capsize=3)
        plt.ylabel(pretty_metric)
        plt.title(f"Forecast stability across origins — {_pretty_cat(cat)}")
        plt.xticks(x, g["model_label"], ha="center")

        # Extend y-axis dynamically to fit labels + error bars
        y_max = float(np.nanmax(values + errors)) if len(values) else 1.0
        plt.ylim(0, y_max * 1.15)

        #label bars with mean MAE values
        for i, v in enumerate(values):
            if np.isnan(v):
                continue
            plt.text(
                x[i],
                v + errors[i] + 0.02 * y_max,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
                )

        plt.tight_layout()
        out = FIG_DIR / f"{cat}_origin_stability_mae.png"
        plt.savefig(out, dpi=300)
        plt.close()

#4. dedicated event window trace figure
#plots actual + rolling average + event ridge + erf over most ative event window in electronics (Q4 or exam)
def fig_event_window_trace(category: str, event_col: str, models: list[str]) -> None:
    series = {}
    for m in models:
        df = _load_detailed_for(category, m)
        df[event_col] = df[event_col].astype(bool)

        #agg per forecast_time: mean y_true, mean y_pred, max event flag
        agg = (
            df.groupby("forecast_time")[["y_true", "y_pred", event_col]]
            .agg({"y_true": "mean", "y_pred": "mean", event_col: "max"})
            .reset_index()
            .sort_values("forecast_time")
        )
        series[m] = agg

    #use y_true from first model (should be same across all)
    any_m = models[0]
    base = series[any_m]
    plt.figure(figsize=(10, 5.0))
    plt.plot(base["forecast_time"], base["y_true"], label="Actual")
    
    for m in models:
        plt.plot(series[m]["forecast_time"], series[m]["y_pred"], label=MODEL_LABELS.get(m, m))

    #shade event weeks
    ev = base[base[event_col]]
    if not ev.empty:
        for t in ev["forecast_time"]:
            plt.axvline(t, linestyle="--", linewidth=0.7)
            
    plt.title(f"Forecast vs actual during {event_col} — {_pretty_cat(category)}")
    plt.ylabel("Demand")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / f"{category}_{event_col}_forecast_trace.png"
    plt.savefig(out, dpi=300)
    plt.close()



def main() -> None:
    _ensure_dirs()

    overall = _load_summary("overall_metrics.csv")
    ev = _load_summary("event_vs_nonevent_metrics.csv")
    stability = _load_summary("stability_metrics.csv")

    fig_overall_mae_bar(overall) #figs 3a-3c
    fig_event_vs_nonevent_mae(ev) #figs 1a-1c
    fig_origin_stability(stability) #figs 4a-4c

    try:
        fig_event_window_trace(
        category="electronic_goods",
        event_col="is_q4_holiday_electronics",
        models=["rolling_average", "event_ridge", "event_random_forest"],
    )
    except FileNotFoundError as e:
        print(f"[WARN] Skipped event window trace (missing detailed file): {e}")
    except KeyError as e:
        print(f"[WARN] Skipped event window trace (missing event column): {e}")

    print(f"[OK] Figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
