"""
Split-conformal prediction intervals using absolute residual quantiles.
"""

import numpy as np
import pandas as pd

USE_CONFORMAL_PREDICTION = True
USE_FIXED_WIDTH_CONFORMAL = True
USE_TRAIN_ALL_MEAN_SCALED = (
    False  # Train error model on all train data, scale to match fixed-width mean
)
SHOW_PLOTS = False


def get_error_bars_split_conformal(
    *, y_true: pd.Series, y_pred: pd.Series, cal_idx, alpha=0.10
):
    cal_idx = pd.Index(cal_idx)
    r = (
        (y_true.loc[cal_idx].astype(float) - y_pred.loc[cal_idx].astype(float))
        .abs()
        .dropna()
        .to_numpy(copy=True)
    )

    r.sort()
    k = int(np.ceil((len(r) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(r))
    q = float(r[k - 1])  # global half-width

    return pd.Series(q, index=y_pred.index, name=f"pi{int((1-alpha)*100)}_halfwidth")
