#Pipeline baselines tests
#Runs against real data for one category and checks 
# forecasts are of right length and include no errors/NaNs

from turtle import pd
import pytest
from src.evaluation.metrics import compute_errors
from src.pipeline.run_baselines import run_baselines_for_category

def test_run_baselines_for_category():
    results = run_baselines_for_category(
        category="fitness_equipment",
        demand_csv_path="data/processed/demand_monthly.csv",
        test_weeks=12
    )

    #check keys present
    for key in ["y_train", "y_test", "sn_forecast", "ra_forecast", "tr_forecast"]:
        assert key in results

    y_test = results["y_test"]
    assert len(results["sn_forecast"]) == len(y_test)
    assert len(results["ra_forecast"]) == len(y_test)
    assert len(results["tr_forecast"]) == len(y_test)

    #check no NaNs in forecasts
    assert not results["sn_forecast"].isna().any()
    assert not results["ra_forecast"].isna().any()
    assert not results["tr_forecast"].isna().any()


#error handling test - invalid category
def test_run_baselines_for_invalid_category():
    with pytest.raises(ValueError):
        run_baselines_for_category(
            category="non_existent_category_123",
            demand_csv_path="data/processed/demand_monthly.csv",
            test_weeks=12
        )



