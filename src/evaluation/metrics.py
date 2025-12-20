from __future__ import annotations

import numpy as np
import pandas as pd

def compute_errors(
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> dict:
    """Compute basic forecast error metrics: MAE, RMSE, MAPE.
    Assume y_true and y_pred are aligned pd.Series."""

    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")
    
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must have the same index.")
    
    y_true_values = y_true.to_numpy(dtype=float)
    y_pred_values = y_pred.to_numpy(dtype=float)

    errors = y_true_values - y_pred_values

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    #MAPE: avoid division by zero
    non_zero = y_true_values != 0
    if non_zero.any():
        mape = np.mean(
            np.abs(errors[non_zero] / y_true_values[non_zero])
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }