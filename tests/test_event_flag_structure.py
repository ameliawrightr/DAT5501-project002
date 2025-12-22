import pandas as pd
from src.ingestion.demand import load_weekly_demand


#test event flags exist and are boolean
def test_event_flags_structure():
    df = load_weekly_demand("data/processed/demand_monthly.csv")
    
    event_cols = [c for c in df.columns if c.startswith("is_")]

    #core events flags
    expected = {
        "is_new_year_fitness",
        "is_back_to_school",
        "is_exam_season",
        "is_q4_holiday_electronics",
    }
    assert expected.issubset(set(event_cols))

    #all event flags are boolean
    for col in expected:
        assert df[col].dropna().isin([0, 1]).all()

#test event flags aligned weekly
def test_event_flags_aligned_weekly():
    df = load_weekly_demand("data/processed/demand_monthly.csv")
    
    #pick cat that exists
    category_sample = df["category"].unique()[0]

    df_cat = df[df["category"] == category_sample].copy()

    df_cat = df_cat.set_index("week_start").asfreq("W-MON")

    event_cols = [c for c in df_cat.columns if c.startswith("is_")]

    for col in event_cols:
        #no missing weeks in event flags
        assert not df_cat[col].isna().any()


#FUTURE IMPLEMENTATION:
#test event counts over small known range
#e.g., know how many week are Q4/backtoschool assert counts
