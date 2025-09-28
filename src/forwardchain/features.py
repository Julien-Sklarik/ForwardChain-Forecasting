import pandas as pd

def add_group_lags_and_rolls(df: pd.DataFrame,
                             id_col: str,
                             date_col: str,
                             target_col: str,
                             lags=(1, 2, 5, 7, 14, 28),
                             roll_windows=(3, 7, 14, 28),
                             min_roll_periods=3) -> pd.DataFrame:
    """
    Build safe per group lags and rolling stats on the shifted target.
    """
    parts = []
    for _, g in df.groupby(id_col, sort=False):
        b = g.sort_values(date_col).copy()
        for k in lags:
            b[f"lag_{k}"] = b[target_col].shift(k)
        prev = b[target_col].shift(1)
        for w in roll_windows:
            b[f"roll_mean_{w}"] = prev.rolling(w, min_periods=min_roll_periods).mean()
            b[f"roll_std_{w}"] = prev.rolling(w, min_periods=min_roll_periods).std()
        b["dow"] = b[date_col].dt.weekday
        b["is_month_start"] = b[date_col].dt.is_month_start.astype(int)
        b["is_month_end"] = b[date_col].dt.is_month_end.astype(int)
        parts.append(b)
    out = pd.concat(parts, axis=0)
    return out

def build_features(df: pd.DataFrame,
                   id_col: str,
                   date_col: str,
                   target_col: str,
                   lags=(1, 2, 5, 7, 14, 28),
                   roll_windows=(3, 7, 14, 28),
                   min_roll_periods=3):
    df2 = add_group_lags_and_rolls(df, id_col, date_col, target_col,
                                   lags=lags, roll_windows=roll_windows,
                                   min_roll_periods=min_roll_periods)
    need = [c for c in df2.columns if c.startswith("lag_") or c.startswith("roll_")]
    df2 = df2.dropna(subset=need).reset_index(drop=True)
    y = df2[target_col].astype(float)
    time_vals = df2[date_col].to_numpy()
    X = df2.drop(columns=[target_col, date_col])
    return X, y, time_vals
