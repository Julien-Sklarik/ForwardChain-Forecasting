from pathlib import Path
import pandas as pd

def load_rossmann(data_dir: str, date_col: str = "Date") -> pd.DataFrame:
    """
    Load and join train.csv with store.csv then sort by Store and Date.
    """
    d = Path(data_dir)
    df = pd.read_csv(d / "train.csv")
    store = pd.read_csv(d / "store.csv")
    out = df.merge(store, on="Store", how="left")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(["Store", date_col]).reset_index(drop=True)
    return out
