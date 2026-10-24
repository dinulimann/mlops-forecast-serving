"""Forecast API. Serves whichever model version holds the `champion` alias in
the MLflow registry and swaps to a new one without a restart.

Every answer is logged (requests, predictions, and the exact feature rows for
a fixed 10% sample of items) so the monitor can check drift and, once actuals
arrive, accuracy.
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
import uuid
import zlib
from contextlib import asynccontextmanager
from datetime import date

import mlflow
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from mlops import features as F
from mlops import model as M

log = logging.getLogger("uvicorn.error")
DSN = os.environ.get("DATABASE_URL", "postgresql://mlops:mlops@localhost:5563/mlops")
POLL_SECONDS = int(os.environ.get("MODEL_POLL_SECONDS", 30))
SAMPLE_EVERY = 10  # log feature rows for items whose id hashes to 0 mod 10

REQUESTS = Counter("forecast_requests_total", "Forecast requests", ["status"])
LATENCY = Histogram("forecast_latency_seconds", "Time to answer a forecast request",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
ROWS = Counter("forecast_rows_total", "Item-days forecast")
VERSION = Gauge("model_version", "Model version currently served")
LOADED_AT = Gauge("model_loaded_timestamp_seconds", "When the served model was loaded")
MISSING = Gauge("feature_missing_ratio", "Share of missing values in the last request", ["feature"])
MEAN_FORECAST = Gauge("forecast_mean_units", "Mean forecast per item-day in the last request")


class Served:
    """The model in use. Replaced as a whole, so a request never sees a half-swapped state."""
    def __init__(self, booster=None, version: int | None = None, tags: dict | None = None):
        self.booster, self.version, self.tags = booster, version, tags or {}


state = Served()
lock = threading.Lock()


def load_champion(force: bool = False) -> bool:
    global state
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(M.MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return False
    if not force and state.version == int(mv.version):
        return False
    with lock:
        booster = mlflow.lightgbm.load_model(f"models:/{M.MODEL_NAME}/{mv.version}")
        state = Served(booster, int(mv.version), dict(mv.tags))
    VERSION.set(state.version)
    LOADED_AT.set(time.time())
    log.info("serving %s v%s", M.MODEL_NAME, state.version)
    return True


def watch() -> None:
    while True:
        time.sleep(POLL_SECONDS)
        try:
            load_champion()
        except Exception:  # registry briefly unreachable: keep serving the current model
            log.exception("model refresh failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_champion()
    except Exception:
        log.exception("no model at startup; will keep trying")
    threading.Thread(target=watch, daemon=True).start()
    yield


app = FastAPI(title="Demand forecast service", lifespan=lifespan)


class ForecastRequest(BaseModel):
    origin: date | None = Field(None, description="Last day with known sales. Default: latest loaded day.")
    ids: list[str] | None = Field(None, description="Item ids such as FOODS_3_090_CA_1. Default: all items.",
                                  max_length=5000)


class Point(BaseModel):
    id: str
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    request_id: uuid.UUID
    model_version: int
    origin: date
    horizon_days: int
    forecasts: list[Point]


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    started = time.perf_counter()
    current = state
    if current.booster is None:
        REQUESTS.labels("unavailable").inc()
        raise HTTPException(503, "no champion model registered yet")
    with psycopg.connect(DSN, autocommit=True) as conn:
        sales_until, prices_until = conn.execute(
            "SELECT (SELECT max(d) FROM sales), (SELECT max(d) FROM prices)").fetchone()
        if req.origin is None:
            origin = sales_until
        else:
            row = conn.execute("SELECT d FROM calendar WHERE date = %s", (req.origin,)).fetchone()
            if not row:
                REQUESTS.labels("invalid").inc()
                raise HTTPException(422, f"{req.origin} is outside the calendar")
            origin = row[0]
        if origin > sales_until:
            REQUESTS.labels("invalid").inc()
            raise HTTPException(409, "sales for the origin day have not arrived yet")
        if origin + F.HORIZON > prices_until:
            REQUESTS.labels("invalid").inc()
            raise HTTPException(409, "the price plan does not cover the forecast window yet")
        if req.ids is not None:
            known = {r[0] for r in conn.execute("SELECT id FROM items WHERE id = ANY(%s)", (req.ids,))}
            unknown = sorted(set(req.ids) - known)
            if unknown:
                REQUESTS.labels("invalid").inc()
                raise HTTPException(404, {"unknown_ids": unknown[:20], "count": len(unknown)})

        rows = M.window(conn, origin, req.ids)
        rows["forecast"] = M.predict(current.booster, rows)
        request_id = uuid.uuid4()
        latency = time.perf_counter() - started
        record(conn, request_id, current.version, origin, rows, latency)
        origin_date = conn.execute("SELECT date FROM calendar WHERE d = %s", (origin,)).fetchone()[0]

    REQUESTS.labels("ok").inc()
    LATENCY.observe(latency)
    ROWS.inc(len(rows))
    MEAN_FORECAST.set(float(rows["forecast"].mean()))
    for c in F.NUMERIC:
        MISSING.labels(c).set(float(rows[c].isna().mean()))
    # Built directly: validating 85k Point objects through pydantic costs more
    # memory and time than the forecast itself. The shape matches ForecastResponse.
    dates = rows["date"].dt.strftime("%Y-%m-%d")
    return JSONResponse({
        "request_id": str(request_id), "model_version": current.version, "origin": str(origin_date),
        "horizon_days": F.HORIZON,
        "forecasts": [{"id": i, "date": d, "forecast": round(float(f), 3)}
                      for i, d, f in zip(rows["id"], dates, rows["forecast"])]})


def record(conn, request_id, version, origin, rows, latency) -> None:
    with conn.transaction():
        conn.execute("INSERT INTO requests (request_id, model_version, origin_d, n_items, latency_ms) "
                     "VALUES (%s, %s, %s, %s, %s)",
                     (request_id, version, origin, rows["id"].nunique(), latency * 1000))
        buf = io.StringIO()
        rows.assign(request_id=str(request_id), model_version=version, origin_d=origin)[
            ["request_id", "model_version", "origin_d", "id", "d", "forecast"]].to_csv(buf, index=False, header=False)
        with conn.cursor().copy("COPY predictions FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buf.getvalue())

        sample = rows[[zlib.crc32(i.encode()) % SAMPLE_EVERY == 0 for i in rows["id"]]]
        if len(sample):
            feats = sample[F.NUMERIC].to_json(orient="records", lines=True, double_precision=15).splitlines()
            with conn.cursor().copy("COPY prediction_features (request_id, origin_d, id, d, features) FROM STDIN") as copy:
                for i, d, js in zip(sample["id"], sample["d"], feats):
                    copy.write_row((str(request_id), origin, i, int(d), js))


@app.get("/health")
def health():
    return {"status": "ok" if state.booster is not None else "no_model", "model_version": state.version}


@app.get("/model")
def model_info():
    if state.booster is None:
        raise HTTPException(503, "no champion model registered yet")
    return {"name": M.MODEL_NAME, "version": state.version, "tags": state.tags, "features": F.FEATURES}


@app.post("/admin/reload")
def reload():
    changed = load_champion()
    return {"changed": changed, "model_version": state.version}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
