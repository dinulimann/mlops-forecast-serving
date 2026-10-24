"""Daily checks on the served model. Writes monitor_results and performance,
and returns whether a retrain is needed and whether it is safe.

    python -m mlops.monitor 2015-06-07

Checks
- data_quality: did the feed deliver complete, plausible data? (no actuals needed)
- input_drift:  do the features the model was just served look like its
                training-time reference? (no actuals needed)
- performance:  once a forecast's 28 days are over, how accurate was it,
                against the seasonal naive and against what the gate expected?
- model_age:    how old is the champion?

Decision: retrain on an accuracy, drift or age alert, but never while a data
quality alert is open, so a broken feed is not baked into the next model.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import date

import mlflow
import pandas as pd
from mlflow import MlflowClient

from . import drift, feed
from . import features as F
from . import model as M

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
PSI_WARN, PSI_ALERT = 0.10, 0.25
PRICE_JUMP = 5.0           # day-over-day price ratio beyond x5 or /5 is implausible
PRICE_JUMP_SHARE = 0.01    # ... on more than 1% of priced items
WAPE_DEGRADED = 1.15       # matured WAPE more than 15% above what the gate measured
# A store-level bias beyond 10% over a forecast's 28 days is an alert, not a
# warning: in the replay v3 under-forecast by 10-14% for 11 weeks while only
# warning; retraining on the first alert would have cut store-level WAPE from
# 12.1% to 7.5% over that period (tools/what_if_bias_rule.py).
BIAS_ALERT = 0.10
AGE_WARN, AGE_ALERT = 56, 91
_reference_cache: dict[int, dict] = {}


@dataclass
class Report:
    run_d: int
    rows: list[tuple] = field(default_factory=list)

    def add(self, check, subject, value, threshold, status, detail=""):
        self.rows.append((self.run_d, check, subject, value, threshold, status, detail))

    def alerts(self, check: str | None = None) -> list[tuple]:
        return [r for r in self.rows if r[5] == "alert" and (check is None or r[1] == check)]


def reference(client: MlflowClient, version: int) -> dict:
    if version not in _reference_cache:
        run_id = client.get_model_version(M.MODEL_NAME, str(version)).tags["run_id"]
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference_profile.json")
        with open(path) as f:
            _reference_cache[version] = json.load(f)
    return _reference_cache[version]


def data_quality(conn, rep: Report, T: int) -> None:
    n_items = conn.execute("SELECT count(*) FROM items").fetchone()[0]
    got = dict(conn.execute("SELECT d, count(*) FROM sales WHERE d BETWEEN %s AND %s GROUP BY d", (T - 6, T)).fetchall())
    missing_days = [d for d in range(T - 6, T + 1) if got.get(d, 0) < n_items]
    rep.add("data_quality", "sales_complete", float(len(missing_days)), 0, "alert" if missing_days else "ok",
            f"days with missing items: {missing_days}" if missing_days else "")
    neg = conn.execute("SELECT count(*) FROM sales WHERE d BETWEEN %s AND %s AND sales < 0", (T - 6, T)).fetchone()[0]
    rep.add("data_quality", "sales_non_negative", float(neg), 0, "alert" if neg else "ok")

    # Every price the next forecast will use, compared day over day per item.
    jumps = conn.execute("""
        WITH p AS (
            SELECT id, d, sell_price, lag(sell_price) OVER (PARTITION BY id ORDER BY d) AS prev
            FROM prices WHERE d BETWEEN %s AND %s)
        SELECT d, count(*) FILTER (WHERE sell_price / prev > %s OR sell_price / prev < 1.0 / %s)::real
                  / NULLIF(count(*), 0)
        FROM p WHERE prev IS NOT NULL AND d > %s GROUP BY d ORDER BY d""",
        (T - 35, T + F.HORIZON, PRICE_JUMP, PRICE_JUMP, T - 28)).fetchall()
    worst_d, worst = max(jumps, key=lambda r: r[1] or 0) if jumps else (None, 0.0)
    bad = [d for d, share in jumps if (share or 0) > PRICE_JUMP_SHARE]
    detail = ""
    if bad:
        first, last = feed.date_of(conn, min(bad)), feed.date_of(conn, max(bad))
        detail = f"implausible day-over-day price changes on {len(bad)} days ({first} .. {last}), worst {worst:.0%} of items"
    rep.add("data_quality", "price_plausible", float(worst or 0), PRICE_JUMP_SHARE,
            "alert" if bad else "ok", detail)


def input_drift(conn, client, rep: Report, T: int) -> None:
    req = conn.execute("SELECT request_id, model_version FROM requests WHERE origin_d = %s "
                       "ORDER BY created_at DESC LIMIT 1", (T,)).fetchone()
    if not req:
        rep.add("input_drift", "served", None, None, "warn", "no forecast was served for this day")
        return
    request_id, version = req
    ref = reference(client, version)
    feats = pd.DataFrame([r[0] for r in conn.execute(
        "SELECT features FROM prediction_features WHERE request_id = %s", (request_id,))])
    for name in drift.MONITORED:
        value = drift.psi(ref[name], feats[name])
        status = "alert" if value > PSI_ALERT else "warn" if value > PSI_WARN else "ok"
        rep.add("input_drift", name, value, PSI_ALERT, status)
    preds = pd.Series([r[0] for r in conn.execute(
        "SELECT forecast FROM predictions WHERE request_id = %s", (request_id,))])
    value = drift.psi(ref["forecast"], preds)
    rep.add("input_drift", "forecast", value, PSI_ALERT,
            "alert" if value > PSI_ALERT else "warn" if value > PSI_WARN else "ok")


def performance(conn, client, rep: Report, T: int) -> None:
    """Scores every served forecast whose window has fully happened."""
    due = conn.execute("""
        SELECT r.request_id, r.origin_d, r.model_version FROM requests r
        LEFT JOIN performance p USING (request_id)
        WHERE p.request_id IS NULL AND r.origin_d + %s <= %s AND r.n_items > 1000
        ORDER BY r.origin_d, r.created_at""", (F.HORIZON, T)).fetchall()
    for request_id, origin, version in due:
        served = pd.DataFrame(conn.execute("SELECT id, d, forecast FROM predictions WHERE request_id = %s",
                                           (request_id,)).fetchall(), columns=["id", "d", "forecast"])
        s = M.score(served, M.actuals(conn, origin + 1, origin + F.HORIZON), M.seasonal_naive(conn, origin))
        tags = client.get_model_version(M.MODEL_NAME, str(version)).tags
        conn.execute("""INSERT INTO performance VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                     (request_id, origin, version, s["wape"], s["store_wape"], s["naive_wape"],
                      s["rel_wape"], s["bias"], float(tags.get("expected_wape", "nan"))))

    # Latest forecast per origin, last two matured origins.
    recent = conn.execute("""
        SELECT DISTINCT ON (p.origin_d) p.origin_d, p.wape, p.expected_wape, p.bias, p.rel_wape
        FROM performance p JOIN requests r USING (request_id)
        ORDER BY p.origin_d DESC, r.created_at DESC""").fetchall()[:2]
    if not recent:
        rep.add("performance", "wape_vs_expected", None, WAPE_DEGRADED, "ok", "no matured forecast yet")
        return
    o, wape, expected, bias, rel = recent[0]
    ratios = [r[1] / r[2] for r in recent if r[2]]
    degraded = len(ratios) == 2 and min(ratios) > WAPE_DEGRADED
    rep.add("performance", "wape_vs_expected", ratios[0] if ratios else None, WAPE_DEGRADED,
            "alert" if degraded else "warn" if ratios and ratios[0] > WAPE_DEGRADED else "ok",
            f"forecast from {feed.date_of(conn, o)}: WAPE {wape:.3f} vs {expected:.3f} expected")
    rep.add("performance", "rel_wape_vs_naive", rel, 1.0, "alert" if rel >= 1 else "ok")
    rep.add("performance", "bias", bias, BIAS_ALERT, "alert" if abs(bias) > BIAS_ALERT else "ok")


def model_age(conn, client, rep: Report, T: int) -> None:
    mv = client.get_model_version_by_alias(M.MODEL_NAME, "champion")
    age = T - int(mv.tags["origin_d"])
    rep.add("model_age", "days", float(age), AGE_ALERT,
            "alert" if age > AGE_ALERT else "warn" if age > AGE_WARN else "ok", f"champion v{mv.version}")


def run(conn, T: int) -> dict:
    client = MlflowClient()
    rep = Report(T)
    data_quality(conn, rep, T)
    input_drift(conn, client, rep, T)
    performance(conn, client, rep, T)
    model_age(conn, client, rep, T)
    run_at = conn.execute("SELECT clock_timestamp()").fetchone()[0]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany("INSERT INTO monitor_results (run_d, run_at, check_name, subject, value, threshold, status, detail) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", [(r[0], run_at, *r[1:]) for r in rep.rows])

    triggers = [f"{r[1]}:{r[2]}" for r in rep.alerts() if r[1] != "data_quality"]
    blocked = [f"{r[2]} ({r[6]})" for r in rep.alerts("data_quality")]
    return {"run_d": T, "retrain": bool(triggers) and not blocked, "triggers": triggers, "blocked_by": blocked,
            "alerts": [f"{r[1]}:{r[2]}" for r in rep.alerts()]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day", type=date.fromisoformat)
    args = ap.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    with feed.connect() as conn:
        print(json.dumps(run(conn, feed.day(conn, args.day)), indent=2))


if __name__ == "__main__":
    main()
