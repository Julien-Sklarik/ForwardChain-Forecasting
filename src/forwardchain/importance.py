import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def perm_importance(model, X_valid, y_valid, feature_names, scoring="neg_mean_absolute_error", n_repeats=5, random_state=42):
    r = permutation_importance(model, X_valid, y_valid, scoring=scoring,
                               n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
    return imp
