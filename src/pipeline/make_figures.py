"""
Figure generation for evaluation.

This script produces the key evaluation figures used in the report.

Figures generated
-----------------
1) Overall model performance (sMAPE)
   - Bar chart of mean sMAPE (%) by model, per category.

2) Event vs non-event performance (sMAPE)
   - Grouped bars of mean sMAPE (%) on event weeks vs non-event weeks, per category.

3) Forecast stability across rolling origins (MAE)
   - Bar chart of origin-level MAE mean with error bars (± std), per category.

4) Visual evaluation during high-volatility periods (holdout trace)
   - Forecast vs actual on the last 52 weeks for the electronics category,
     with the Q4 holiday window shaded and multiple models overlaid.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ingestion.demand import load_weekly_demand
from src.models.baseline import rolling_average
from src.models.forecasting_utils import align_predictions
from src.models.event_models import EventModelConfig, fit_and_forecast_event_model

#PATHS/CONSTANTS
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

#Return report friendly category label
def _pretty_cat(cat: str) -> str:
    return CATEGORY_LABELS.get(cat, CATEGORY_LABELS.get(cat, cat))

#Create output directory for figures
def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

#Load summary CSV produced by backtest summarisation
def _load_summary(name: str) -> pd.DataFrame:
    path = SUMMARY_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run summarise_backtests first.")
    return pd.read_csv(path)

#Load detailed rolling origin backtest CSV for a given category and model
def _load_detailed_for(category: str, model: str) -> pd.DataFrame:
    path = BACKTEST_DIR / f"{category}_{model}_detailed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing detailed CSV: {path}")
    df = pd.read_csv(path)
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    return df

#FIGURE 1: Overall error by model (sMAPE)
#Bar chart: mean sMAPE (%) by model for each category
def fig_overall_sMAPE_bar(overall: pd.DataFrame) -> None:
    metric = "sMAPE"
    pretty_metric = "Mean sMAPE (%)"

    #plot all core models (baseline + event-aware)
    key_models = MODEL_ORDER

    for cat, g in overall.groupby("category"):
        g = g[g["model"].isin(key_models)].copy()
        if g.empty:
            continue 

        #consistent order (by performance) 
        g = g.set_index("model").reindex(MODEL_ORDER).reset_index()
        g = g.dropna(subset=[metric])
        g["model_label"] = g["model"].map(MODEL_LABELS).fillna(g["model"])

        x = np.arange(len(g), dtype=float)
        values = g[metric].astype(float).values
        
        plt.figure(figsize=(6, 5.0))
        plt.bar(x, values)
        plt.ylabel(pretty_metric)
        plt.title(f"Overall {pretty_metric} by model — {_pretty_cat(cat)}")
        plt.xticks(x, g["model_label"], ha="center", fontsize=8)

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

#FIGURE 2: Event vs non-event error comparison (sMAPE)
#Grouped bars: mean sMAPE (%) on event weeks vs non-event weeks
def fig_event_vs_nonevent_sMAPE(ev: pd.DataFrame) -> None:
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

        plt.figure(figsize=(6, 5.0))
        plt.bar(x - width / 2, vals_event, width, label="Event weeks")
        plt.bar(x + width / 2, vals_nonevent, width, label="Non-event weeks")
        
        plt.ylabel(pretty_metric)
        plt.title(f"{pretty_metric}: event vs non-event — {_pretty_cat(cat)}")
        plt.xticks(x, [MODEL_LABELS.get(m, m) for m in wide.index], ha="center", fontsize=8)
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

#FIGURE 3: Stability plot across origins (MAE mean +- std)
#Bar chart: origin level MAE mean with error bars (± std), per category
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

        plt.figure(figsize=(6, 5.0))
        plt.bar(x, values, yerr=errors, capsize=3)
        plt.ylabel(pretty_metric)
        plt.title(f"Forecast stability across origins — {_pretty_cat(cat)}")
        plt.xticks(x, g["model_label"], ha="center", fontsize=8)

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


#FIGURE HELPER: shade boolean event spans 
#Shade contiguous spans where mask is True
def _shade_event_spans(ax, idx: pd.DatetimeIndex, mask: pd.Series, label:str) -> None:
    mask = mask.reindex(idx).fillna(False).astype(bool)
    
    in_span = False
    span_start = None
    prev = None
    first_label_used = False

    for t, is_ev in mask.items():
        if is_ev and not in_span:
            in_span = True
            span_start = t
        if (not is_ev) and in_span:
            ax.axvspan(
                span_start,
                prev,
                alpha=0.15,
                label=(label if not first_label_used else None),
            )
            first_label_used = True
            in_span = False
        prev = t

    #close an open span at the end of series
    if in_span and span_start is not None and prev is not None:
        ax.axvspan(
            span_start,
            prev,
            alpha=0.15,
            label=(label if not first_label_used else None),
        )

#FIGURE 4: Electronics holdout trace with Q4 shading
#Forecast vs actual on last 52 weeks for electronics, with Q4 holiday window shaded
def make_electronics_volatility_figure(
        demand_csv_path: str = "data/processed/demand_monthly.csv",
        category: str = "electronic_goods",
        test_weeks: int = 52,
        rolling_window: int = 12,
        train_weeks_to_show: int = 26,
    ) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_weekly_demand(demand_csv_path)
    df_cat = df[df["category"] == category].copy()
    if df_cat.empty:
        raise ValueError(f"No demand data found for category {category}")
    
    df_cat = df_cat.sort_values("week_start").set_index("week_start")

    y = df_cat["demand"].astype(float).sort_index()

    event_cols = [c for c in df_cat.columns if c.startswith("is_")]
    if not event_cols:
        raise ValueError(f"No event columns found for category {category}")
    events = df_cat[event_cols].asfreq("W-MON").fillna(False).astype(bool)

    y_train = y.iloc[:-test_weeks]
    y_test = y.iloc[-test_weeks:]
    events_train = events.loc[y_train.index]
    events_test = events.loc[y_test.index]

    #baseline: RA on holdout window
    ra_raw = rolling_average(y_train, window=rolling_window, horizon=len(y_test))
    ra_forecast = align_predictions(ra_raw, index=y_test.index)

    # --- Event models: same configs as run_event_models ---
    model_configs: Dict[str, EventModelConfig] = {
        "Event Ridge": EventModelConfig(
            model_type="ridge",
            lags=(1, 52),
            include_trend=True,
            include_weekofyear=True,
            alpha=1.0,
        ),
        "Random Forest (event-aware)": EventModelConfig(
            model_type="random_forest",
            lags=(1, 52),
            include_trend=True,
            include_weekofyear=True,
            rf_n_estimators=300,
            rf_max_depth=None,
        ),
    }

    event_forecasts: Dict[str, pd.Series] = {}
    for name, config in model_configs.items():
        y_pred, _ = fit_and_forecast_event_model(
            y_train=y_train,
            y_test=y_test,
            events_train=events_train,
            events_test=events_test,
            config=config,
        )
        event_forecasts[name] = y_pred.reindex(y_test.index)

    # --- Plot: last N train weeks + test window with all forecasts ---
    train_tail = y_train.iloc[-train_weeks_to_show:]

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Train context
    ax.plot(
        train_tail.index,
        train_tail.values,
        label=f"Train (last {train_weeks_to_show} weeks)",
        color="0.8",
        linewidth=1.2,
    )

    # Actual test
    ax.plot(
        y_test.index,
        y_test.values,
        label="Test / Actual",
        color="black",
        linewidth=2.0,
    )

    # Rolling Average baseline
    ax.plot(
        y_test.index,
        ra_forecast.values,
        label="Rolling Average (baseline)",
        linewidth=1.6,
        linestyle="--",
    )

    # Event-aware models
    for name, pred in event_forecasts.items():
        ax.plot(
            pred.index,
            pred.values,
            label=name,
            linewidth=1.6,
        )

    # Shade Q4 holiday window for electronics
    if "is_q4_holiday_electronics" in events_test.columns:
        _shade_event_spans(
            ax=ax,
            idx=y_test.index,
            mask=events_test["is_q4_holiday_electronics"],
            label="Q4 holiday window",
        )

    ax.axvline(y_test.index.min(), linestyle="--", alpha=0.6, color="grey", linewidth=0.8)
    ax.text(y_test.index.min(), ax.get_ylim()[1], " Test start", va="top", ha="left", fontsize=8, color="grey")
    ax.set_title("Holdout forecasts (52-week horizon) — Electronic Accessories (Q4 highlighted)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Demand (weekly)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(train_tail.index.min(), y_test.index.max())
    fig.tight_layout()

    out_path = FIG_DIR / "figure2_electronic_holdout_q4.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Wrote {out_path}")


#Generate all figs required for report
def main() -> None:
    _ensure_dirs()

    overall = _load_summary("overall_metrics.csv")
    ev = _load_summary("event_vs_nonevent_metrics.csv")
    stability = _load_summary("stability_metrics.csv")

    fig_overall_sMAPE_bar(overall) 
    fig_event_vs_nonevent_sMAPE(ev) 
    fig_origin_stability(stability) 
    make_electronics_volatility_figure() 

if __name__ == "__main__":
    main()
