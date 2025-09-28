import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forward_oof(pipe, X, y, time_vals, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y), dtype=float)
    rows = []
    for k, (tr, te) in enumerate(tscv.split(X), start=1):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        rows.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_valid": int(len(te)),
            "valid_start": pd.to_datetime(time_vals[te].min()).isoformat(),
            "valid_end": pd.to_datetime(time_vals[te].max()).isoformat(),
            "MAE": float(mean_absolute_error(y.iloc[te], pred)),
            "RMSE": float(mean_squared_error(y.iloc[te], pred, squared=False)),
        })
    return oof, pd.DataFrame(rows)

def holdout_last_split(pipe, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    tr, te = splits[-1]
    m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    return float(mean_absolute_error(y.iloc[te], pred)), float(
        mean_squared_error(y.iloc[te], pred, squared=False)
    )
