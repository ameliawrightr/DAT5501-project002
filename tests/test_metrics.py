import pandas as pd
from src.evaluation.metrics import compute_errors

def test_compute_errors_perfect_forecast():
    idx = pd.date_range(start="2020-01-01", periods=5, freq="W-MON")
    y_true = pd.Series([10, 20, 30, 40, 50], index=idx)
    y_pred = y_true.copy()

    errors = compute_errors(y_true, y_pred)

    # Check that all expected error metrics are present
    expected_metrics = {"MAE", "RMSE", "sMAPE"}
    assert expected_metrics.issubset(errors.keys())

    # Validate the computed error values
    assert errors["MAE"] == 0
    assert errors["RMSE"] == 0
    assert errors["sMAPE"] == 0

def test_compute_errors_constant_bias():
    idx = pd.date_range(start="2020-01-01", periods=3, freq="W-MON")
    y_true = pd.Series([10, 20, 30], index=idx)
    y_pred = pd.Series([15, 25, 35], index=idx)  # Constant bias of +5

    errors = compute_errors(y_true, y_pred)

    # MAE should be ~5
    assert abs(errors["MAE"] - 5) < 1e-6
    # RMSE should be ~5
    assert abs(errors["RMSE"] - 5) < 1e-6
    # sMAPE should be > 0
    assert errors["sMAPE"] > 0

#metrics test - sMAPE behaviour on simple known vectors
def test_smape_behavior():
    idx = pd.date_range(start='2020-01-01', periods=2, freq='D')
    y_true = pd.Series([50 , 100], index=idx)
    y_pred = pd.Series([100, 50], index=idx) 

    errors = compute_errors(y_true, y_pred)
    #sMAPE should be between 0 and 200 + finite
    assert 0 <= errors['sMAPE'] <= 200


