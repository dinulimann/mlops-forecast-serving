"""Assertions after a short replay in CI (see .github/workflows/ci.yml)."""
import sys

import httpx

from mlops import feed

conn = feed.connect()
q = lambda sql, *a: conn.execute(sql, a).fetchall()
incident = sys.argv[1]
d = feed.day(conn, incident)

weeks = q("SELECT count(DISTINCT origin_d) FROM requests")[0][0]
assert weeks >= 8, f"only {weeks} weeks served"

runs = [st for (st,) in q("SELECT status FROM monitor_results WHERE run_d = %s AND subject = 'price_plausible' "
                         "ORDER BY run_at", d)]
assert runs == ["alert", "ok"], f"expected an alert, then a clean run after the fix: {runs}"
alerts = q("SELECT count(*) FROM requests WHERE origin_d = %s", d)[0][0]
assert alerts == 2, "incident week should have the broken forecast and the re-issued one"
redelivered = q("SELECT count(*) FROM feed_batches WHERE note LIKE 're-delivery%%'")[0][0]
assert redelivered == 1

scored = q("SELECT count(*), avg(rel_wape) FROM performance")[0]
assert scored[0] >= 4 and scored[1] < 1, f"served forecasts should beat the seasonal naive: {scored}"

r = httpx.post("http://localhost:8090/forecast", json={"ids": ["FOODS_3_090_CA_1"]}, timeout=60)
assert r.status_code == 200 and len(r.json()["forecasts"]) == 28, r.text
assert httpx.post("http://localhost:8090/forecast", json={"ids": ["NOPE_CA_1"]}).status_code == 404
metrics = httpx.get("http://localhost:8090/metrics").text
assert 'forecast_requests_total{status="ok"}' in metrics
print(f"ok: {weeks} weeks served, {scored[0]} forecasts scored (mean WAPE vs naive {scored[1]:.2f}), incident detected and fixed")
