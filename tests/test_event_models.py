#Test event_models.py
#Tes:
# - feature matrix shape + columns
# - fit+forecast shape + index
# - test model learns event uplif

import pandas as pd
import numpy as np

from src.models.event_models import (
    EventModelConfig,
    build_autoreg_event_training_matrix,
    fit_event_model,
    fit_and_forecast_event_model
)

def synthetic_series_with_events():
    #synthetic weekly data where event weeks have high demand
    idx_train = pd.date_range("2020-01-01", periods=20, freq="W-MON")
    idx_test = pd.date_range("2020-05-11", periods=4, freq="W-MON")

    #event: first 10 weeks False, next 10 True
    is_event_train = pd.Series([False] * 10 + [True] * 10, index=idx_train)
    #test events: alternating pattern
    is_event_test = pd.Series([False, True, False, True], index=idx_test)

    #demand: base 10, +5 uplift on event weeks
    y_train = pd.Series(10 + 5 * is_event_train.astype(int), index=idx_train)
    y_test = pd.Series(10 + 5 * is_event_test.astype(int), index=idx_test)

    events_train = pd.DataFrame({'is_event': is_event_train.astype(int)}, index=idx_train)
    events_test = pd.DataFrame({'is_event': is_event_test.astype(int)}, index=idx_test)

    return y_train, y_test, events_train, events_test, is_event_test

def test_build_autoreg_event_training_matrix():
    y_train, _, events_train, _, _ = synthetic_series_with_events()

    #config with no lags and no time features → pure event feature
    cfg = EventModelConfig(
        model_type="ridge",
        lags=(),                 # no autoregressive lags
        include_trend=False,
        include_weekofyear=False,
    )

    X, y_target = build_autoreg_event_training_matrix(
        y=y_train,
        events=events_train,
        config=cfg,
    )

    #with no lags and no time features, X should only contain event column
    assert list(X.columns) == ["is_event"]
    assert len(X) == len(y_target) == len(y_train)
    #no NaNs
    assert not X.isna().any().any()
    assert not y_target.isna().any()

def test_fit_and_forecast_event_model_shape_and_index():
    y_train, y_test, events_train, events_test, _ = synthetic_series_with_events()

    cfg = EventModelConfig(
        model_type="ridge",
        lags=(),                 # no autoregressive lags
        include_trend=False,
        include_weekofyear=False,
    )

    forecasts, result = fit_and_forecast_event_model(
        y_train=y_train,
        y_test=y_test,
        events_train=events_train,
        events_test=events_test,
        config=cfg,
    )

    #correct type and length
    assert isinstance(forecasts, pd.Series)
    assert len(forecasts) == len(y_test)

    #aligned index
    assert list(forecasts.index) == list(y_test.index)

    #no NaNs
    assert not forecasts.isna().any()

    #result should be an EventModelForecast with trained model
    assert result.n_train_points == len(y_train)
    assert result.max_lag == 0  # because we set lags=()


def test_event_model_learns_event_uplift_on_training_data():
    y_train, _, events_train, _, _ = synthetic_series_with_events()

    cfg = EventModelConfig(
        model_type="ridge",
        lags=(),                 # no autoregressive lags
        include_trend=False,
        include_weekofyear=False,
    )

    #fit model
    result = fit_event_model(
        y_train=y_train,
        events_train=events_train,
        config=cfg,
    )

    #rebuild training design matrix using same config
    X_train, y_target = build_autoreg_event_training_matrix(
        y=y_train,
        events=events_train,
        config=cfg,
    )

    #predictions on training design matrix
    y_hat = result.model.predict(X_train.values)
    y_hat = pd.Series(y_hat, index=X_train.index)

    #compare predicted means for event vs non-event rows
    is_event = X_train["is_event"] == 1
    mean_event = y_hat[is_event].mean()
    mean_non_event = y_hat[~is_event].mean()

    #Oon this synthetic data, events should be predicted higher
    assert mean_event > mean_non_event