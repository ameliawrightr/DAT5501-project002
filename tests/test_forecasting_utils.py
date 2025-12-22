#test align_predictions directly - as rely to fix indices
import pandas as pd
from src.models.forecasting_utils import align_predictions

def test_align_predictions():
    #raw forecast with defaults RangeIndex
    raw = pd.series([1,2,3])
    target_index = pd.date_range(start='2023-01-01', periods=3, freq='W-MON')

    aligned = align_predictions(raw, index=target_index)

    assert list(aligned.index) == list(target_index)
    assert list(aligned.values) == [1,2,3]