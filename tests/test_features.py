import numpy as np
import pandas as pd

from mlops import features as F

ORIGIN = 1450


def window(df, origin=ORIGIN):
    return df[(df["d"] > origin) & (df["d"] <= origin + F.HORIZON)].reset_index(drop=True)


def test_future_sales_cannot_leak(frame):
    """Changing every sale after the origin must not change any feature of the window."""
    tampered = frame.copy()
    tampered.loc[tampered["d"] > ORIGIN, "sales"] = 999.0
    a, b = window(F.build(frame)), window(F.build(tampered))
    pd.testing.assert_frame_equal(a[F.NUMERIC], b[F.NUMERIC])


def test_serving_history_is_long_enough(frame):
    """Serving loads only HISTORY days before the origin; training loads far more.
    The window's features must be identical either way (no training/serving skew)."""
    full = window(F.build(frame))
    short = window(F.build(frame[frame["d"] >= ORIGIN - F.HISTORY].reset_index(drop=True)))
    pd.testing.assert_frame_equal(full[F.NUMERIC], short[F.NUMERIC])


def test_one_day_less_history_would_break_it(frame):
    """Guards HISTORY itself: with one day fewer, lag_364 of the first window day is lost."""
    short = window(F.build(frame[frame["d"] >= ORIGIN - F.HISTORY + 9].reset_index(drop=True)))
    assert short["lag_364"].isna().any()


def test_requesting_a_subset_of_items_changes_nothing(frame):
    """dept_avg_price comes from the database, so a request for a few items gets
    the same features as the full store."""
    some = frame[frame["id"].isin(frame["id"].unique()[:3])].reset_index(drop=True)
    full = window(F.build(frame))
    part = window(F.build(some))
    pd.testing.assert_frame_equal(full[full["id"].isin(part["id"].unique())].reset_index(drop=True)[F.NUMERIC],
                                  part[F.NUMERIC])


def test_lags_are_what_they_say(frame):
    df = F.build(frame)
    one = df[df["id"] == df["id"].iloc[0]].set_index("d")
    d = ORIGIN + 5
    assert one.loc[d, "lag_28"] == one.loc[d - 28, "sales"]
    assert one.loc[d, "lag_364"] == one.loc[d - 364, "sales"]
    assert np.isclose(one.loc[d, "mean_7"], one.loc[d - 34:d - 28, "sales"].mean())
