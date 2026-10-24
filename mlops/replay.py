"""Replays a year of operations week by week against the running stack:

    every Sunday T:  feed delivers last week's sales and the price plan to T+28
                     -> client asks the API for a 28-day forecast from T
                     -> monitor runs; on an alert it may retrain (gate + promote)
                     -> on a data quality alert the "operator" gets the feed re-delivered
                        and the forecast re-issued

    python -m mlops.replay --start 2015-05-31 --end 2016-05-22 --incident 2015-10-04:FOODS_3

The incident: in that week's delivery the price plan of one department
arrives in cents instead of dollars (x100), a classic upstream unit bug.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, timedelta

import httpx
import mlflow

from . import feed, monitor, train
from . import features as F

API = os.environ.get("API_URL", "http://localhost:8090")
COOLDOWN_DAYS = 28  # a retrain is gated on the last 28 days, so retraining more often adds nothing


def ask(origin: date, attempts: int = 5) -> dict:
    """Asks for a forecast; if the service is down (restarting), waits and retries."""
    for attempt in range(attempts):
        try:
            r = httpx.post(f"{API}/forecast", json={"origin": str(origin)}, timeout=120)
            r.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == attempts - 1:
                raise
            print(json.dumps({"week": str(origin), "retry": f"{type(e).__name__}: {e}"[:200]}), flush=True)
            for _ in range(60):
                time.sleep(5)
                try:
                    if httpx.get(f"{API}/health", timeout=5).json().get("model_version"):
                        break
                except httpx.TransportError:
                    pass
    body = r.json()
    return {"request_id": body["request_id"], "model_version": body["model_version"], "rows": len(body["forecasts"])}


def last_training(conn) -> int | None:
    return conn.execute("SELECT max(at_d) FROM model_events WHERE event IN ('promoted', 'rejected')").fetchone()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2015, 5, 31))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2016, 5, 22))
    ap.add_argument("--incident", default="2015-10-04:FOODS_3", help="DATE:DEPT, or empty for none")
    args = ap.parse_args()
    mlflow.set_tracking_uri(train.TRACKING)
    incident_day, incident_dept = (args.incident.split(":") + [None])[:2] if args.incident else (None, None)

    with feed.connect() as conn:
        feed.init(conn)
        start = feed.day(conn, args.start)
        if feed.loaded_until(conn) is None:
            feed.advance(conn, start)
        if train.champion(mlflow.MlflowClient()) is None:
            print(json.dumps(train.run(conn, start, "initial")))
            httpx.post(f"{API}/admin/reload", timeout=120)

        T = args.start
        while T <= args.end:
            t0 = time.time()
            d = feed.day(conn, T)
            if conn.execute("SELECT 1 FROM monitor_results WHERE run_d = %s LIMIT 1", (d,)).fetchone():
                T += timedelta(days=7)  # already done in an earlier, interrupted run
                continue
            broken = incident_day and T == date.fromisoformat(incident_day)
            if d > (feed.loaded_until(conn) or 0):
                feed.advance(conn, d, price_factor={incident_dept: 100.0} if broken else None,
                             note=f"INCIDENT: {incident_dept} prices in cents" if broken else None)
            served = ask(T)
            decision = monitor.run(conn, d)
            line = {"week": str(T), "model": served["model_version"], "alerts": decision["alerts"]}

            if decision["blocked_by"]:
                # Operator response: upstream re-delivers the affected price days, then the
                # forecast is re-issued and the monitor re-run.
                bad_from = conn.execute("SELECT d_from, d_to FROM feed_batches WHERE kind = 'prices' "
                                        "ORDER BY batch_id DESC LIMIT 1").fetchone()
                feed.load(conn, "prices", *bad_from, note="re-delivery after data quality alert")
                served = ask(T)
                decision = monitor.run(conn, d)
                line["action"] = "prices re-delivered, forecast re-issued"
                line["after_fix"] = decision["alerts"]

            last = last_training(conn)
            if decision["retrain"] and (last is None or d - last >= COOLDOWN_DAYS):
                result = train.run(conn, d, ",".join(decision["triggers"]))
                if result["decision"] == "promoted":
                    httpx.post(f"{API}/admin/reload", timeout=120).raise_for_status()
                line["retrain"] = {k: result.get(k) for k in ("decision", "version", "reason")}
                line["retrain"]["eval_wape"] = round(result["eval_wape"], 4)
                if "champion_eval_wape" in result:
                    line["retrain"]["champion_eval_wape"] = round(result["champion_eval_wape"], 4)
            elif decision["retrain"]:
                line["retrain"] = f"wanted ({','.join(decision['triggers'])}) but in cooldown"
            line["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(line), flush=True)
            T += timedelta(days=7)

        # Let the last forecasts mature: deliver the remaining actuals and score them.
        last_d = conn.execute("SELECT max(d) FROM calendar").fetchone()[0]
        feed.advance(conn, last_d)
        monitor.performance(conn, mlflow.MlflowClient(), monitor.Report(last_d), last_d)
        print(json.dumps({"done": True, "scored_through": str(feed.date_of(conn, last_d))}))


if __name__ == "__main__":
    main()
