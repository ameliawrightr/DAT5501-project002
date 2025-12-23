#OUTPUT CONCISE CSVs SUMMARISING BACKTEST RESULTS
#1. overall_metrics.csv 
#2. event_vs_nonevent_metrics.csv
#3. stability_metrics.csv

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

BACKTESTS_DIR = Path("artifacts/backtests/")
OUTPUT_DIR = Path("artifacts/summary/")

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred))
    out = np.zeros_like(denominator, dtype=float)
    mask = denominator != 0
    out[mask] = 2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(np.mean(out) * 100)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def load_all_detailed() -> pd.DataFrame:
    """Load all detailed backtest results into a single DataFrame."""
    files = sorted(glob.glob(str(BACKTESTS_DIR / "*_detailed.csv")))
    
    dfs = [] 
    for f in files:
        df = pd.read_csv(f)
        #extract model name from filename
        name = Path(f).name
        category = name.split("_")[0]
        if "category" not in df.columns:
            df["category"] = category
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    #parse timestamps
    for col in ["origin_time", "forecast_time"]:
        if col in all_df.columns:
            all_df[col] = pd.to_datetime(all_df[col])

    return all_df

def add_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add event vs non-event labels based on event_start and event_end columns."""
    event_cols = [c for c in df.columns if c.startswith("is_")]
    if not event_cols:
        df["is_event_week"] = False
        df["event_type"] = "none"
        return df
    
    #ensure bool
    for c in event_cols:
        df[c] = df[c].astype(bool)

    df["is_event_week"] = df[event_cols].any(axis=1)

    def _label(row) -> str:
        trues = [c for c in event_cols if bool(row[c])]
        if len(trues) == 0:
            return "none"
        if len(trues) == 1:
            return trues[0]
        return "multiple"

    df["event_type"] = df.apply(_label, axis=1)

    return df

def summarise_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Generate overall metrics summary DataFrame."""
    rows = []
    for (cat, model), g in df.groupby(["category", "model"]):
        y_true = g["y_true"].to_numpy(float)
        y_pred = g["y_pred"].to_numpy(float)

        rows.append(
            {
                "category": cat,
                "model": model,
                "RMSE": rmse(y_true, y_pred),
                "sMAPE": smape(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "n_points": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values(["category", "MAE"])

def summarise_event_vs_nonevent(df: pd.DataFrame) -> pd.DataFrame:
    """Generate event vs non-event metrics summary DataFrame."""
    rows = []
    for (cat, model, is_event), g in df.groupby(["category", "model", "is_event_week"]):
        y_true = g["y_true"].to_numpy(float)
        y_pred = g["y_pred"].to_numpy(float)

        rows.append(
            {
                "category": cat,
                "model": model,
                "subset": "event_weeks" if is_event else "non_event_weeks",
                "RMSE": rmse(y_true, y_pred),
                "sMAPE": smape(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "n_points": int(len(g)),
            }
        )

    return pd.DataFrame(rows).sort_values(["category", "subset", "MAE"])

def summarise_stability(df: pd.DataFrame) -> pd.DataFrame:
    # stability across origins: compute per-origin MAE, then summarise dispersion
    rows = []
    for (cat, model, origin), g in df.groupby(["category", "model", "origin_time"]):
        y_true = g["y_true"].to_numpy(float)
        y_pred = g["y_pred"].to_numpy(float)
        rows.append(
            {
                "category": cat,
                "model": model,
                "origin_time": origin,
                "origin_MAE": mae(y_true, y_pred),
            }
        )
    per_origin = pd.DataFrame(rows)

    out_rows = []
    for (cat, model), g in per_origin.groupby(["category", "model"]):
        vals = g["origin_MAE"].to_numpy(float)
        out_rows.append(
            {
                "category": cat,
                "model": model,
                "origin_MAE_mean": float(np.mean(vals)),
                "origin_MAE_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "origin_MAE_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) > 1 else 0.0,
                "n_origins": int(len(g)),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["category", "origin_MAE_mean"])


def summarise_by_event_type(df: pd.DataFrame) -> pd.DataFrame:
    # Breakdown by specific event flag (useful for Discussion)
    rows = []
    for (cat, model, etype), g in df.groupby(["category", "model", "event_type"]):
        if etype == "none":
            continue
        y_true = g["y_true"].to_numpy(float)
        y_pred = g["y_pred"].to_numpy(float)
        rows.append(
            {
                "category": cat,
                "model": model,
                "event_type": etype,
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "sMAPE": smape(y_true, y_pred),
                "n_points": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "event_type", "MAE"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_detailed()
    df = add_event_labels(df)

    overall = summarise_overall(df)
    ev_nonev = summarise_event_vs_nonevent(df)
    stability = summarise_stability(df)
    by_event = summarise_by_event_type(df)

    overall.to_csv(OUTPUT_DIR / "overall_metrics.csv", index=False)
    ev_nonev.to_csv(OUTPUT_DIR / "event_vs_nonevent_metrics.csv", index=False)
    stability.to_csv(OUTPUT_DIR / "stability_metrics.csv", index=False)
    if not by_event.empty:
        by_event.to_csv(OUTPUT_DIR / "event_breakdown_by_flag.csv", index=False)

    print(f"[OK] Wrote: {OUTPUT_DIR / 'overall_metrics.csv'}")
    print(f"[OK] Wrote: {OUTPUT_DIR / 'event_vs_nonevent_metrics.csv'}")
    print(f"[OK] Wrote: {OUTPUT_DIR / 'stability_metrics.csv'}")
    if not by_event.empty:
        print(f"[OK] Wrote: {OUTPUT_DIR / 'event_breakdown_by_flag.csv'}")


if __name__ == "__main__":
    main()