import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def test_forward_split_is_temporal():
    t = pd.date_range("2020-01-01", periods=100, freq="D")
    X = pd.DataFrame({"x": range(100), "Date": t})
    y = pd.Series(range(100))
    tscv = TimeSeriesSplit(n_splits=5)
    for tr, te in tscv.split(X):
        assert X["Date"].iloc[tr].max() < X["Date"].iloc[te].min()
