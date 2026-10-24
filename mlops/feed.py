"""Plays the role of the upstream systems: sales actuals arrive after each day,
the price plan is published 28 days ahead.

    python -m mlops.feed init 2015-05-31     # schema + history up to that day
    python -m mlops.feed advance 2015-06-07  # one more week of actuals and prices
"""
from __future__ import annotations

import argparse
import io
import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg

from .features import HORIZON

DATA = Path(os.environ.get("DATA_DIR", "data"))
DSN = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5563/mlops")
SCHEMA = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True)


def day(conn, when: date | str) -> int:
    row = conn.execute("SELECT d FROM calendar WHERE date = %s", (str(when),)).fetchone()
    if not row:
        raise ValueError(f"{when} is not in the calendar")
    return row[0]


def date_of(conn, d: int) -> date:
    return conn.execute("SELECT date FROM calendar WHERE d = %s", (d,)).fetchone()[0]


def _copy(conn, table: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    with conn.cursor().copy(f"COPY {table} ({', '.join(df.columns)}) FROM STDIN WITH (FORMAT csv)") as copy:
        copy.write(buf.getvalue())


def init(conn, until: int | None = None) -> None:
    conn.execute(SCHEMA.read_text())
    if conn.execute("SELECT count(*) FROM calendar").fetchone()[0] == 0:
        cal = pd.read_parquet(DATA / "calendar.parquet")
        cal["date"] = cal["date"].dt.date
        _copy(conn, "calendar", cal)
        _copy(conn, "items", pd.read_parquet(DATA / "items.parquet"))


def loaded_until(conn) -> int | None:
    return conn.execute("SELECT max(d_to) FROM feed_batches WHERE kind = 'sales'").fetchone()[0]


def load(conn, kind: str, d_from: int, d_to: int, price_factor: dict[str, float] | None = None,
         note: str | None = None) -> int:
    """Replaces [d_from, d_to] of `sales` or `prices` with the source data.
    price_factor {dept_id: factor} reproduces a broken delivery."""
    df = pd.read_parquet(DATA / f"{kind}.parquet", filters=[("d", ">=", d_from), ("d", "<=", d_to)])
    if price_factor:
        items = pd.read_parquet(DATA / "items.parquet").set_index("id")["dept_id"]
        factor = df["id"].map(items).map(price_factor).fillna(1.0)
        df["sell_price"] = (df["sell_price"] * factor).round(2)
    with conn.transaction():
        conn.execute(f"DELETE FROM {kind} WHERE d BETWEEN %s AND %s", (d_from, d_to))
        _copy(conn, kind, df)
        conn.execute("INSERT INTO feed_batches (kind, d_from, d_to, rows, note) VALUES (%s, %s, %s, %s, %s)",
                     (kind, d_from, d_to, len(df), note))
    return len(df)


def advance(conn, to_d: int, price_factor: dict[str, float] | None = None, note: str | None = None) -> None:
    """Actuals up to to_d, price plan up to to_d + HORIZON."""
    start = (loaded_until(conn) or conn.execute("SELECT min(d) - 1 FROM calendar").fetchone()[0]) + 1
    last = conn.execute("SELECT max(d) FROM calendar").fetchone()[0]
    if to_d >= start:
        load(conn, "sales", start, to_d)
    p_from = start + HORIZON if start > conn.execute("SELECT min(d) FROM calendar").fetchone()[0] else start
    p_to = min(to_d + HORIZON, last)
    if p_to >= p_from:
        load(conn, "prices", p_from, p_to, price_factor, note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["init", "advance"])
    ap.add_argument("until", type=date.fromisoformat)
    args = ap.parse_args()
    with connect() as conn:
        if args.action == "init":
            init(conn)
        advance(conn, day(conn, args.until))
        print(f"data loaded up to {args.until} (prices to +{HORIZON} days)")


if __name__ == "__main__":
    main()
