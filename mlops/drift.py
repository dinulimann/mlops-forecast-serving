"""Population stability index against a reference profile saved with each model."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Calendar features (dow, dom, month, events) and `age` (days since an item's
# first sale, which grows every week) change by design and are not monitored:
# the first replay alerted on `age` every single week. These are the inputs
# whose distribution should be stable unless something changed upstream or in demand.
MONITORED = ["sell_price", "price_vs_max", "price_change", "price_vs_dept",
             "lag_28", "lag_364", "mean_7", "mean_28", "mean_182", "std_28", "zero_share_56"]
BINS = 10
EPS = 1e-4
DECIMALS = 6  # values are rounded before binning: features are float32 in training,
              # float64 after the JSON round trip, and many sit exactly on a bin edge (k/7)


def profile(rows: pd.DataFrame, forecast: np.ndarray | None = None) -> dict:
    """Decile edges and shares per feature (plus the forecast itself). Missing
    values get a bin of their own, so a feed that starts sending NULLs shows up."""
    cols = {c: rows[c].astype(float) for c in MONITORED}
    if forecast is not None:
        cols["forecast"] = pd.Series(forecast, dtype=float)
    out = {}
    for name, s in cols.items():
        v = s.dropna().to_numpy().round(DECIMALS)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1)[1:-1]).round(DECIMALS)) if len(v) else np.array([])
        out[name] = {"edges": edges.tolist(), "shares": _shares(s, edges).tolist()}
    return out


def _shares(s: pd.Series, edges: np.ndarray) -> np.ndarray:
    v = s.to_numpy(dtype=float).round(DECIMALS)
    missing = np.isnan(v)
    counts = np.bincount(np.searchsorted(edges, v[~missing], side="right"), minlength=len(edges) + 1)
    counts = np.append(counts, missing.sum()).astype(float)
    return counts / max(len(v), 1)


def psi(reference: dict, current: pd.Series) -> float:
    edges = np.asarray(reference["edges"])
    ref = np.clip(np.asarray(reference["shares"]), EPS, None)
    cur = np.clip(_shares(current.astype(float), edges), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
