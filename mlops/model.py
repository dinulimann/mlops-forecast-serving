"""Training, forecasting and scoring. Shared by the training job, the API and
the monitor."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F

PARAMS = {
    "objective": "tweedie", "tweedie_variance_power": 1.1,  # many zeros, long right tail
    "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 0.1, "verbose": -1, "seed": 42, "num_threads": 0,
}
TRAIN_DAYS = 730
MODEL_NAME = "m5_ca1_forecast"


def first_d(conn) -> int:
    return conn.execute("SELECT min(d) FROM calendar").fetchone()[0]


def fit(conn, origin: int, rounds: int = 600) -> tuple[lgb.Booster, int]:
    start = max(first_d(conn), origin - TRAIN_DAYS - F.HISTORY)
    df = F.build(F.load_frame(conn, origin, start, origin))
    train = df[(df["d"] > origin - TRAIN_DAYS) & df["lag_56"].notna() & df["sales"].notna()]
    data = lgb.Dataset(train[F.FEATURES], train["sales"], categorical_feature=F.CATEGORICAL)
    return lgb.train(PARAMS, data, num_boost_round=rounds), len(train)


def window(conn, origin: int, ids: list[str] | None = None) -> pd.DataFrame:
    """Feature rows of the forecast window origin+1 .. origin+HORIZON."""
    df = F.build(F.load_frame(conn, origin, origin - F.HISTORY, origin + F.HORIZON, ids))
    return df[df["d"] > origin].reset_index(drop=True)


def predict(booster: lgb.Booster, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(booster.predict(rows[F.FEATURES]), 0, None)


def seasonal_naive(conn, origin: int) -> pd.DataFrame:
    """Repeat the last observed week: the baseline every forecast must beat."""
    last = pd.DataFrame(conn.execute(
        "SELECT id, d, sales FROM sales WHERE d BETWEEN %s AND %s", (origin - 6, origin)).fetchall(),
        columns=["id", "src", "forecast"])
    days = pd.DataFrame({"d": np.arange(origin + 1, origin + F.HORIZON + 1)})
    days["src"] = origin - 6 + (days["d"] - origin - 1) % 7
    return days.merge(last, on="src")[["id", "d", "forecast"]].astype({"forecast": float})


def actuals(conn, d_from: int, d_to: int) -> pd.DataFrame:
    return pd.DataFrame(conn.execute("SELECT id, d, sales FROM sales WHERE d BETWEEN %s AND %s",
                                     (d_from, d_to)).fetchall(), columns=["id", "d", "actual"])


def score(forecast: pd.DataFrame, actual: pd.DataFrame, naive: pd.DataFrame | None = None) -> dict:
    """WAPE at item-day level (what a replenishment system consumes), WAPE of
    the store's daily total (what planning looks at), bias, and the item-level
    WAPE relative to the seasonal naive (< 1 = better than the baseline)."""
    m = forecast.merge(actual, on=["id", "d"], how="inner")
    if m.empty:
        raise ValueError("no overlapping actuals to score")
    total = m["actual"].sum()
    out = {
        "wape": float((m["forecast"] - m["actual"]).abs().sum() / total),
        "bias": float((m["forecast"].sum() - total) / total),
        "rows": int(len(m)),
    }
    daily = m.groupby("d")[["forecast", "actual"]].sum()
    out["store_wape"] = float((daily["forecast"] - daily["actual"]).abs().sum() / daily["actual"].sum())
    if naive is not None:
        n = naive.merge(actual, on=["id", "d"])
        out["naive_wape"] = float((n["forecast"] - n["actual"]).abs().sum() / n["actual"].sum())
        out["rel_wape"] = out["wape"] / out["naive_wape"]
    return out
