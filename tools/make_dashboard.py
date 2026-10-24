"""Generates the two Grafana dashboards (kept in code so the panels are
reviewable as SQL and PromQL rather than as JSON blobs):

- forecast-quality.json: accuracy, drift, alerts and model decisions, plotted
  by business date (2015-2016 in the replay)
- forecast-service.json: the live API from Prometheus, last 30 minutes
"""
import json
from pathlib import Path

PG = {"type": "grafana-postgresql-datasource", "uid": "pg"}
PROM = {"type": "prometheus", "uid": "prom"}
LATEST = """(SELECT DISTINCT ON (p.origin_d) p.* FROM performance p JOIN requests r USING (request_id)
           ORDER BY p.origin_d, r.created_at DESC)"""
panels, y = [], 0


def save(uid, title, time, refresh=""):
    global panels, y
    dash = {"uid": uid, "title": title, "schemaVersion": 41, "editable": True,
            "time": time, "refresh": refresh, "panels": panels}
    Path(f"grafana/dashboards/{uid}.json").write_text(json.dumps(dash, indent=2))
    print(f"{uid}: {len(panels)} panels")
    panels, y = [], 0


def panel(title, kind, targets, w=12, h=8, x=0, unit=None, desc="", overrides=None, live=False, **opts):
    p = {"type": kind, "title": title, "description": desc, "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "datasource": PROM if live else PG, "targets": targets,
         "fieldConfig": {"defaults": {"unit": unit} if unit else {}, "overrides": overrides or []},
         "options": opts.pop("options", {})}
    p.update(opts)
    panels.append(p)


def sql(q, fmt="time_series", ref="A"):
    return {"refId": ref, "datasource": PG, "rawQuery": True, "editorMode": "code", "format": fmt, "rawSql": q}


def prom(expr, legend, ref="A"):
    return {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend}


def row(title):
    global y
    panels.append({"type": "row", "title": title, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "collapsed": False})
    y += 1


row("Accuracy of served forecasts (scored once their 28 days are over)")
panel("Item-level WAPE by forecast origin", "timeseries", [sql(f"""
SELECT c.date::timestamp AS time, p.wape AS "served model", p.naive_wape AS "seasonal naive",
       p.expected_wape AS "expected at promotion"
FROM {LATEST} p JOIN calendar c ON c.d = p.origin_d ORDER BY 1""")], unit="percentunit",
      desc="Lower is better. Each point is the forecast issued that Sunday, scored on the 28 days that followed.")
panel("Store total: daily WAPE and bias", "timeseries", [sql(f"""
SELECT c.date::timestamp AS time, p.store_wape AS "store-level WAPE", p.bias AS "bias"
FROM {LATEST} p JOIN calendar c ON c.d = p.origin_d ORDER BY 1""")], x=12, unit="percentunit")
y += 8

row("Monitor")
panel("Input drift (PSI vs the champion's training reference)", "timeseries", [sql("""
SELECT DISTINCT ON (m.run_d, m.subject) c.date::timestamp AS time, m.subject AS metric, m.value
FROM monitor_results m JOIN calendar c ON c.d = m.run_d
WHERE m.check_name = 'input_drift' AND m.value IS NOT NULL ORDER BY m.run_d, m.subject, m.run_at""")], w=16, h=9,
      desc="PSI above 0.25 is an alert; first monitor run of each day. Calendar features and age are not monitored: they change by design.",
      options={"legend": {"displayMode": "table", "placement": "right"}})
panel("Alerts and model decisions", "table", [sql("""
SELECT c.date::text AS day, e.event AS what, coalesce('v' || e.model_version, '') AS model, e.reason AS detail
FROM model_events e JOIN calendar c ON c.d = e.at_d
UNION ALL
SELECT c.date::text, 'alert', m.check_name || ': ' || m.subject,
       coalesce(nullif(m.detail, ''), round(m.value::numeric, 3)::text)
FROM monitor_results m JOIN calendar c ON c.d = m.run_d WHERE m.status = 'alert'
ORDER BY 1 DESC, 2""", fmt="table")], x=16, w=8, h=9,
      overrides=[{"matcher": {"id": "byName", "options": "day"}, "properties": [{"id": "custom.width", "value": 100}]},
                 {"matcher": {"id": "byName", "options": "what"}, "properties": [{"id": "custom.width", "value": 80}]},
                 {"matcher": {"id": "byName", "options": "model"}, "properties": [{"id": "custom.width", "value": 170}]}])
y += 9

save("forecast-quality", "Forecast quality", {"from": "2015-05-25T00:00:00Z", "to": "2016-06-26T00:00:00Z"})

row("Live service (Prometheus)")
panel("Requests per minute", "timeseries", [prom('sum by (status) (rate(forecast_requests_total[1m])) * 60', "{{status}}")],
      w=6, h=7, live=True)
panel("Latency p50 / p95", "timeseries", [prom('histogram_quantile(0.5, sum by (le) (rate(forecast_latency_seconds_bucket[5m])))', "p50"),
                                           prom('histogram_quantile(0.95, sum by (le) (rate(forecast_latency_seconds_bucket[5m])))', "p95", "B")],
      x=6, w=6, unit="s", live=True)
panel("Model version served", "stat", [prom("model_version", "version")], x=12, w=4, live=True)
panel("Missing feature values, last request", "bargauge", [prom("topk(6, feature_missing_ratio)", "{{feature}}")],
      x=16, w=8, unit="percentunit", live=True, options={"displayMode": "basic", "orientation": "horizontal"})

save("forecast-service", "Forecast service (live)", {"from": "now-30m", "to": "now"}, refresh="10s")
