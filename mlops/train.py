"""Train a challenger, gate it against the champion, register and promote it.

    python -m mlops.train 2015-05-31 [--reason scheduled]

Gate: the challenger is first trained with data up to origin-28 and scored on
the last 28 days, the same window the champion forecast from origin-28 (its
predictions are read back from the predictions table, i.e. what was actually
served). It is promoted only if it beats the seasonal naive and is not worse
than the champion by more than 1%. A promoted model is then refit on all data
up to the origin, so it does not lose four weeks of recency.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from . import drift, feed
from . import features as F
from . import model as M

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
TRACKING = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT = "m5-ca1-forecast"
MAX_LOSS_VS_CHAMPION = 0.01
ROUNDS = int(os.environ.get("TRAIN_ROUNDS", 600))


def champion(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(M.MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return None


def served(conn, origin: int) -> pd.DataFrame | None:
    """The forecast the service returned for this origin (latest request)."""
    rows = conn.execute("""
        SELECT p.id, p.d, p.forecast, p.model_version FROM predictions p
        WHERE p.request_id = (SELECT request_id FROM requests WHERE origin_d = %s
                              ORDER BY created_at DESC LIMIT 1)""", (origin,)).fetchall()
    return pd.DataFrame(rows, columns=["id", "d", "forecast", "model_version"]) if rows else None


def event(conn, at_d: int, kind: str, version: int | None, reason: str, metrics: dict) -> None:
    conn.execute("INSERT INTO model_events (at_d, event, model_version, reason, metrics) VALUES (%s, %s, %s, %s, %s)",
                 (at_d, kind, version, reason, json.dumps(metrics)))


def run(conn, origin: int, reason: str = "manual", rounds: int = ROUNDS) -> dict:
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    current = champion(client)
    when = feed.date_of(conn, origin)

    with mlflow.start_run(run_name=f"origin {when}") as run:
        mlflow.set_tags({"origin": str(when), "trigger": reason})
        mlflow.log_params({**{k: v for k, v in M.PARAMS.items() if k != "verbose"},
                           "rounds": rounds, "train_days": M.TRAIN_DAYS, "origin_d": origin})

        # 1. gate on the most recent 28 days that have actuals
        eval_origin = origin - F.HORIZON
        booster, _ = M.fit(conn, eval_origin, rounds)
        rows = M.window(conn, eval_origin)
        challenger = rows[["id", "d"]].assign(forecast=M.predict(booster, rows))
        actual = M.actuals(conn, eval_origin + 1, origin)
        naive = M.seasonal_naive(conn, eval_origin)
        metrics = {f"eval_{k}": v for k, v in M.score(challenger, actual, naive).items()}
        prior = served(conn, eval_origin)
        if prior is not None:
            champ = M.score(prior, actual)
            metrics.update(champion_eval_wape=champ["wape"], champion_eval_store_wape=champ["store_wape"])
        mlflow.log_metrics(metrics)

        beats_naive = metrics["eval_rel_wape"] < 1
        beats_champion = prior is None or metrics["eval_wape"] <= metrics["champion_eval_wape"] * (1 + MAX_LOSS_VS_CHAMPION)
        if not (beats_naive and beats_champion):
            why = "worse than seasonal naive" if not beats_naive else \
                f"worse than champion v{current.version if current else '?'} on the last 28 days"
            mlflow.set_tag("decision", "rejected")
            event(conn, origin, "rejected", None, f"{reason}: {why}", metrics)
            return {"decision": "rejected", "reason": why, **metrics}

        # 2. refit on everything up to the origin and register
        booster, n_rows = M.fit(conn, origin, rounds)
        rows = M.window(conn, origin)
        forecast = M.predict(booster, rows)
        reference = drift.profile(rows, forecast)
        mlflow.log_metric("train_rows", n_rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference_profile.json"
            path.write_text(json.dumps(reference))
            mlflow.log_artifact(str(path))
            imp = pd.Series(booster.feature_importance("gain"), index=F.FEATURES).sort_values(ascending=False)
            (Path(tmp) / "feature_importance.csv").write_text(imp.to_csv(header=["gain"]))
            mlflow.log_artifact(str(Path(tmp) / "feature_importance.csv"))
        info = mlflow.lightgbm.log_model(booster, name="model", registered_model_name=M.MODEL_NAME)
        version = int(info.registered_model_version)
        client.set_model_version_tag(M.MODEL_NAME, str(version), "origin", str(when))
        client.set_model_version_tag(M.MODEL_NAME, str(version), "origin_d", str(origin))
        client.set_model_version_tag(M.MODEL_NAME, str(version), "expected_wape", f"{metrics['eval_wape']:.4f}")
        client.set_model_version_tag(M.MODEL_NAME, str(version), "run_id", run.info.run_id)
        client.set_registered_model_alias(M.MODEL_NAME, "champion", str(version))
        mlflow.set_tag("decision", "promoted")
        event(conn, origin, "promoted", version, reason, metrics)
        return {"decision": "promoted", "version": version, **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("origin", type=date.fromisoformat)
    ap.add_argument("--reason", default="manual")
    args = ap.parse_args()
    with feed.connect() as conn:
        print(json.dumps(run(conn, feed.day(conn, args.origin), args.reason), indent=2))


if __name__ == "__main__":
    main()
