import json
from pathlib import Path
import yaml
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from forwardchain.data import load_rossmann
from forwardchain.features import build_features
from forwardchain.model import build_preprocessor, build_model_pipeline
from forwardchain.cv import forward_oof, holdout_last_split
from forwardchain.importance import perm_importance
from forwardchain.plots import plot_top_importances
import matplotlib.pyplot as plt
import pandas as pd

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    data_dir = cfg["data_dir"]
    id_col = cfg["id_col"]
    date_col = cfg["date_col"]
    target_col = cfg["target_col"]

    df = load_rossmann(data_dir, date_col=date_col)
    X0, y, time_vals = build_features(df, id_col=id_col, date_col=date_col, target_col=target_col,
                                      lags=cfg["lags"], roll_windows=cfg["roll_windows"],
                                      min_roll_periods=cfg["min_roll_periods"])
    pre, cols = build_preprocessor(X0, high_cardinality_threshold=cfg["high_cardinality_threshold"])
    pipe = build_model_pipeline(pre,
                                learning_rate=cfg["model"]["learning_rate"],
                                max_depth=cfg["model"]["max_depth"],
                                min_samples_leaf=cfg["model"]["min_samples_leaf"])

    param_grid = {
        "model__learning_rate": cfg["tuning"]["learning_rate"],
        "model__max_depth": cfg["tuning"]["max_depth"],
    }
    # Use forward aware CV
    tscv = TimeSeriesSplit(n_splits=cfg["n_splits"])
    gs = GridSearchCV(pipe, param_grid, cv=tscv,
                      scoring="neg_mean_absolute_error", n_jobs=1, verbose=1)
    gs.fit(X0, y)
    best = gs.best_estimator_

    oof, fold_df = forward_oof(best, X0, y, time_vals, n_splits=cfg["n_splits"])
    hold_mae, hold_rmse = holdout_last_split(best, X0, y, n_splits=cfg["n_splits"])

    results_dir = Path("results"); results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("figures"); figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "oof_mae_mean": float(fold_df["MAE"].mean()),
        "oof_mae_std": float(fold_df["MAE"].std()),
        "oof_rmse_mean": float(fold_df["RMSE"].mean()),
        "oof_rmse_std": float(fold_df["RMSE"].std()),
        "holdout_mae": float(hold_mae),
        "holdout_rmse": float(hold_rmse),
        "best_params": gs.best_params_,
        "n_rows": int(X0.shape[0]),
        "n_features": int(X0.shape[1]),
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Permutation importance on the last fold holdout
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cfg["n_splits"])
    tr, te = list(tscv.split(X0))[-1]
    m_last = best.fit(X0.iloc[tr], y.iloc[tr])
    imp = perm_importance(m_last, X0.iloc[te], y.iloc[te], feature_names=cols)
    imp.to_csv(results_dir / "feature_importance.csv", header=["delta_mae"])

    plot_top_importances(imp, k=10, title="Top permutation importances")
    plt.savefig(figures_dir / "top10_importances.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved results to", results_dir, "and", figures_dir)

if __name__ == "__main__":
    main()
