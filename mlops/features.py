"""Features for a direct 28-day forecast, and the one function that loads the
rows they are built from. Training and the API both call `load_frame` +
`build`, so what the model is trained on and what it is served are computed
by the same code from the same tables (tests/test_skew.py checks they agree).

Every sales-based feature is lagged by at least HORIZON days, so all 28 days
of a forecast are built from data known at the origin.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

HORIZON = 28
LAGS = [28, 35, 42, 49, 56, 364]
WINDOWS = [7, 28, 56, 182]
# Days of history needed before the origin so every feature of the forecast
# window is exact: lag_364 of origin+1 reaches back to origin-363, the
# trailing 364-day price max to origin-362.
HISTORY = 371
CATEGORICAL = ["item_id", "dept_id", "cat_id", "event_name_1", "event_type_1"]
NUMERIC = (["snap_ca", "sell_price", "price_vs_max", "price_change", "price_vs_dept",
            "dow", "dom", "month", "age", "std_28", "zero_share_56"]
           + [f"lag_{l}" for l in LAGS] + [f"mean_{w}" for w in WINDOWS])
FEATURES = CATEGORICAL + NUMERIC

FRAME_SQL = """
WITH dp AS (
    SELECT i.store_id, i.dept_id, p.d, avg(p.sell_price)::real AS dept_avg_price
    FROM prices p JOIN items i USING (id)
    WHERE p.d BETWEEN {d_from} AND {d_to}
    GROUP BY 1, 2, 3
), first_sale AS (
    SELECT id, min(d) AS first_sale_d FROM sales WHERE sales > 0 AND d <= {origin} GROUP BY id
)
SELECT i.id, i.item_id, i.dept_id, i.cat_id, c.d, c.date, c.event_name_1, c.event_type_1, c.snap_ca,
       CASE WHEN c.d <= {origin} THEN s.sales END AS sales,
       p.sell_price, dp.dept_avg_price, f.first_sale_d
FROM items i
CROSS JOIN calendar c
LEFT JOIN sales s ON s.id = i.id AND s.d = c.d
LEFT JOIN prices p ON p.id = i.id AND p.d = c.d
LEFT JOIN dp ON dp.store_id = i.store_id AND dp.dept_id = i.dept_id AND dp.d = c.d
LEFT JOIN first_sale f ON f.id = i.id
WHERE c.d BETWEEN {d_from} AND {d_to} {id_filter}
ORDER BY i.id, c.d
"""


def load_frame(conn, origin: int, d_from: int, d_to: int, ids: list[str] | None = None) -> pd.DataFrame:
    """One row per item and day in [d_from, d_to], as known at `origin`:
    sales after the origin are NULL even if they are already in the table."""
    id_filter = ""
    if ids is not None:
        quoted = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
        id_filter = f"AND i.id IN ({quoted})"
    sql = FRAME_SQL.format(origin=int(origin), d_from=int(d_from), d_to=int(d_to), id_filter=id_filter)
    buf = io.BytesIO()
    with conn.cursor().copy(f"COPY ({sql}) TO STDOUT WITH (FORMAT csv, HEADER)") as copy:
        for chunk in copy:
            buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf, engine="pyarrow", parse_dates=["date"], dtype={
        "id": "string", "item_id": "string", "dept_id": "string", "cat_id": "string",
        "event_name_1": "string", "event_type_1": "string", "sales": "float32",
        "sell_price": "float32", "dept_avg_price": "float32", "first_sale_d": "float32"})


def _rolling(s: pd.Series, ids: pd.Series, window: int, how: str, min_periods: int = 1) -> pd.Series:
    r = s.groupby(ids, sort=False).rolling(window, min_periods=min_periods)
    return getattr(r, how)().reset_index(level=0, drop=True).astype(np.float32)


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """frame: output of load_frame, sorted by id and d, every day present."""
    df = frame.copy()
    ids = df["id"]
    g = df.groupby(ids, sort=False)["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag).astype(np.float32)
    shifted = g.shift(HORIZON)
    for w in WINDOWS:
        df[f"mean_{w}"] = _rolling(shifted, ids, w, "mean")
    df["std_28"] = _rolling(shifted, ids, 28, "std", min_periods=2)
    df["zero_share_56"] = _rolling(shifted.eq(0).astype(np.float32).where(shifted.notna()), ids, 56, "mean")

    df["dow"] = df["date"].dt.day_of_week.astype(np.int8)
    df["dom"] = df["date"].dt.day.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["age"] = (df["d"] - df["first_sale_d"]).astype(np.float32)

    # Prices are published ahead, so they are known for the whole window.
    price = df["sell_price"]
    df["price_vs_max"] = (price / _rolling(price, ids, 364, "max")).astype(np.float32)
    df["price_change"] = (price / df.groupby(ids, sort=False)["sell_price"].shift(7) - 1).astype(np.float32)
    df["price_vs_dept"] = (price / df["dept_avg_price"]).astype(np.float32)

    for c in CATEGORICAL:
        df[c] = df[c].fillna("none").astype("category")
    df["snap_ca"] = df["snap_ca"].astype(np.int8)
    return df
