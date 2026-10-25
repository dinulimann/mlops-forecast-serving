"""What if sustained bias had triggered a retrain? (offline, after a replay)

In the replay the monitor warned about under-forecasting from 2016-01-10 on,
but warnings do not trigger anything; the next retrain came from the age rule
on 2016-03-20. This trains the model the bias rule would have produced on
2016-01-10 (same gate, same code) and scores it on the origins that v3
actually served until v4 took over.

    python tools/what_if_bias_rule.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from the repo root

from mlops import feed
from mlops import features as F
from mlops import model as M

conn = feed.connect()
trigger = feed.day(conn, "2016-01-10")
until = feed.day(conn, "2016-03-20")

# The same gate train.py applies: fit up to trigger-28, score the last 28 days
# against what the champion actually served for that window.
booster, _ = M.fit(conn, trigger - F.HORIZON)
rows = M.window(conn, trigger - F.HORIZON)
gate = M.score(rows[["id", "d"]].assign(forecast=M.predict(booster, rows)),
               M.actuals(conn, trigger - F.HORIZON + 1, trigger))
served = pd.DataFrame(conn.execute("""
    SELECT p.id, p.d, p.forecast FROM predictions p WHERE p.request_id = (
        SELECT request_id FROM requests WHERE origin_d = %s ORDER BY created_at DESC LIMIT 1)""",
    (trigger - F.HORIZON,)).fetchall(), columns=["id", "d", "forecast"])
champ = M.score(served, M.actuals(conn, trigger - F.HORIZON + 1, trigger))
print(json.dumps({"gate": {"challenger_wape": gate["wape"], "champion_wape": champ["wape"],
                           "challenger_store_wape": gate["store_wape"], "champion_store_wape": champ["store_wape"]}}))

booster, _ = M.fit(conn, trigger)  # the refit that would have been promoted
out = []
for origin in range(trigger, until, 7):
    rows = M.window(conn, origin)
    s = M.score(rows[["id", "d"]].assign(forecast=M.predict(booster, rows)),
                M.actuals(conn, origin + 1, origin + F.HORIZON))
    v3 = conn.execute("""SELECT DISTINCT ON (p.origin_d) p.wape, p.store_wape, p.bias
                         FROM performance p JOIN requests r USING (request_id)
                         WHERE p.origin_d = %s ORDER BY p.origin_d, r.created_at DESC""", (origin,)).fetchone()
    out.append({"origin": str(feed.date_of(conn, origin)), "what_if_wape": s["wape"], "served_wape": v3[0],
                "what_if_store_wape": s["store_wape"], "served_store_wape": v3[1],
                "what_if_bias": s["bias"], "served_bias": v3[2]})
df = pd.DataFrame(out)
df.to_csv("docs/what_if_bias_rule.csv", index=False, float_format="%.4f")
print(df.round(3).to_string(index=False))
print(df.drop(columns="origin").mean().round(4).to_string())
