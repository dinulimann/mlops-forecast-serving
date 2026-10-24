import json

import numpy as np
import pandas as pd

from mlops import drift
from mlops.monitor import PSI_ALERT


def rows(rng, n=20000, scale=1.0):
    df = pd.DataFrame({c: rng.gamma(1.0, 2.0, n) for c in drift.MONITORED})
    df["mean_7"] = rng.integers(0, 30, n) / 7          # discrete values that sit on bin edges
    df["sell_price"] = rng.uniform(1, 20, n) * scale
    return df.astype("float32")


def test_same_distribution_after_json_round_trip_has_no_drift():
    """The service logs features as JSON; float32 -> JSON -> float64 must not
    move values across bin edges (it did before values were rounded)."""
    rng = np.random.default_rng(1)
    ref = drift.profile(rows(rng))
    served = rows(rng, 3000)
    logged = pd.DataFrame([json.loads(line) for line in
                           served.to_json(orient="records", lines=True).splitlines()])
    for c in drift.MONITORED:
        assert drift.psi(ref[c], logged[c]) < 0.02, c


def test_prices_in_cents_are_an_alert():
    rng = np.random.default_rng(2)
    ref = drift.profile(rows(rng))
    assert drift.psi(ref["sell_price"], rows(rng, 3000, scale=100)["sell_price"]) > 1


def test_a_feed_that_starts_sending_nulls_is_visible():
    rng = np.random.default_rng(3)
    ref = drift.profile(rows(rng))
    cur = rows(rng, 3000)
    cur.loc[cur.index[:900], "sell_price"] = np.nan    # 30% missing
    assert drift.psi(ref["sell_price"], cur["sell_price"]) > PSI_ALERT
