import pandas as pd
from src.models.baseline import (
    seasonal_naive,
    rolling_average,
    time_regression,
)

def make_simple_weekly_series():
    idx = pd.date_range(start="2020-01-05", periods=10, freq="W-MON")
    #simple incresing deamnd
    s = pd.Series(range(10), index=idx)
    return s

def test_seasonal_naive():
    y = make_simple_weekly_series()
    seasonal_period = 4
    horizon = 5

    forecast = seasonal_naive(y, seasonal_period=seasonal_period, horizon=horizon)

    assert len(forecast) == horizon
    #forecast index should follow directly after last training point
    expected_index = pd.date_range(y.index[-1] + pd.offsets.Week(1), periods=horizon, freq="W-MON")
    assert list(forecast.index) == list(expected_index)

def test_rolling_average():
    y = make_simple_weekly_series()
    window = 3
    horizon = 4

    forecast = rolling_average(y, window=window, horizon=horizon)

    assert len(forecast) == horizon
    expected_index = pd.date_range(y.index[-1] + pd.offsets.Week(1), periods=horizon, freq="W-MON")
    assert list(forecast.index) == list(expected_index)

def test_time_regression():
    y = make_simple_weekly_series()
    horizon = 3

    forecast, _ = time_regression(history=y, horizon=horizon)

    assert len(forecast) == horizon
    #no NaNs
    assert not forecast.isna().any()
    
