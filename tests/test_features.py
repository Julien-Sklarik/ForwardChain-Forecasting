import pandas as pd
from forwardchain.features import add_group_lags_and_rolls

def test_lag_alignment():
    df = pd.DataFrame({
        "Store": [1]*5 + [2]*5,
        "Date": pd.date_range("2020-01-01", periods=5).tolist() * 2,
        "Sales": [10, 11, 12, 13, 14, 20, 22, 21, 23, 25]
    })
    out = add_group_lags_and_rolls(df, id_col="Store", date_col="Date", target_col="Sales",
                                   lags=(1,), roll_windows=(3,), min_roll_periods=1)
    g1 = out[out["Store"] == 1].sort_values("Date")
    assert g1["lag_1"].iloc[1] == g1["Sales"].iloc[0]
