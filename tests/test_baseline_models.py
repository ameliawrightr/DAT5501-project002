import pandas as pd
from src.models.baseline import (
    seasonal_naive,
    rolling_average,
    time_regression,
)

def make_simple_weekly_series():
    idx = pd.date_range(start="2020-01-06", periods=10, freq="W-MON")
    #simple incresing deamnd
    s = pd.Series(range(10), index=idx)
    return s

def test_seasonal_naive():
    y = make_simple_weekly_series()
    seasonal_period = 4
    horizon = 5

    forecast = seasonal_naive(y, seasonal_period=seasonal_period, horizon=horizon)

    #correct length
    assert len(forecast) == horizon
    #no NaNs
    assert not forecast.isna().any()

def test_rolling_average():
    y = make_simple_weekly_series()
    window = 3
    horizon = 4

    forecast = rolling_average(y, window=window, horizon=horizon)

    #correct length
    assert len(forecast) == horizon
    #no NaNs
    assert not forecast.isna().any()

def test_time_regression():
    y = make_simple_weekly_series()
    horizon = 3

    forecast, _ = time_regression(history=y, horizon=horizon)

    #correct length
    assert len(forecast) == horizon
    #no NaNs
    assert not forecast.isna().any()
    
