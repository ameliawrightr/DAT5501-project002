from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

def compute_errors(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute basic forecast error metrics for two aligned time series: 
    Return: MAE, RMSE, sMAPE

    Requirements/assumptions:
     - Inputs must be pandas Series
     - Indices must match exactly 
     - NaN pairs removed before computation
     - Assume y_true and y_pred are aligned pd.Series

     Parameters: y_true : pd.Series (ground truth observations)
                 y_pred : pd.Series (model predictions aligned to y_true)
    """
    #type and alignment validation
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")
    
    #exact index equality avoids silent misalignment
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must have the same index.")
    
    #convert to numpy arrays for computation
    y_true_values = y_true.to_numpy(dtype=float)
    y_pred_values = y_pred.to_numpy(dtype=float)

    #pairwise NaN filtering: only keep positions where both values are present
    mask = (~np.isnan(y_true_values)) & (~np.isnan(y_pred_values))
    if mask.sum() == 0:
        #no comparable data points: return NaN for all metrics
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    
    y_t = y_true_values[mask]
    y_p = y_pred_values[mask]

    #core error vector
    errors = y_t - y_p

    #MAE/RMSE
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    #sMAPE: symmetric MAPE to avoid division by zero issues
    denom = np.abs(y_t) + np.abs(y_p)
    non_zero_denom = denom != 0
    
    if non_zero_denom.any():
        smape = np.mean(200 * np.abs(errors[non_zero_denom]) / denom[non_zero_denom])
    else:
        smape = np.nan

    return {"MAE": mae,"RMSE": rmse, "sMAPE": smape}