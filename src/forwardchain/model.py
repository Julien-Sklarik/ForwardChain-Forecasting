import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

def build_preprocessor(X, high_cardinality_threshold=255):
    hi_card = [c for c in X.select_dtypes(exclude="number").columns
               if X[c].nunique(dropna=False) > high_cardinality_threshold]
    X = X.drop(columns=hi_card) if hi_card else X
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1, encoded_missing_value=-2))])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)],
                            remainder="drop")
    return pre, list(num_cols) + list(cat_cols)

def build_model_pipeline(pre, learning_rate=0.1, max_depth=None, min_samples_leaf=20):
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    return Pipeline([("pre", pre), ("model", model)])
