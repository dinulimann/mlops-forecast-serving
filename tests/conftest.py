import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def frame():
    """Synthetic load_frame output: 30 items x 500 days, every day present."""
    rng = np.random.default_rng(0)
    n_items, days = 30, np.arange(1000, 1500)
    ids = [f"FOODS_1_{i:03d}_CA_1" for i in range(n_items)]
    df = pd.DataFrame({"id": np.repeat(ids, len(days)), "d": np.tile(days, n_items)})
    df["item_id"] = df["id"].str[:-5]
    df["dept_id"], df["cat_id"] = "FOODS_1", "FOODS"
    df["date"] = pd.Timestamp("2013-10-06") + pd.to_timedelta(df["d"] - 1000, unit="D")
    df["event_name_1"] = np.where(df["d"] % 50 == 0, "SuperBowl", None)
    df["event_type_1"] = np.where(df["d"] % 50 == 0, "Sporting", None)
    df["snap_ca"] = (df["d"] % 3 == 0).astype(int)
    rate = np.repeat(rng.gamma(1.0, 2.0, n_items), len(days))
    df["sales"] = rng.poisson(rate).astype("float32")
    df["sell_price"] = np.repeat(rng.uniform(1, 20, n_items), len(days)).astype("float32")
    df.loc[df["d"] % 90 == 0, "sell_price"] *= 0.8
    df["dept_avg_price"] = df.groupby("d")["sell_price"].transform("mean")
    first = df[df["sales"] > 0].groupby("id")["d"].min()
    df["first_sale_d"] = df["id"].map(first).astype("float32")
    for c in ["id", "item_id", "dept_id", "cat_id", "event_name_1", "event_type_1"]:
        df[c] = df[c].astype("string")
    return df
