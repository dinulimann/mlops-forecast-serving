"""After a replay: how did the monitored, retrained service do compared with
never retraining (the first model frozen) and with the seasonal naive?

    python -m mlops.report      # writes docs/results.csv, docs/accuracy.png, prints a summary

The frozen model is re-run offline on the same origins with the same feature
code; the served numbers are what the API actually returned.
"""
from __future__ import annotations

import json

import matplotlib
import mlflow
import pandas as pd

from . import feed, train
from . import model as M

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    mlflow.set_tracking_uri(train.TRACKING)
    conn = feed.connect()
    served = pd.DataFrame(conn.execute("""
        SELECT DISTINCT ON (p.origin_d) p.origin_d, p.model_version, p.wape, p.store_wape, p.naive_wape, p.bias
        FROM performance p JOIN requests r USING (request_id) ORDER BY p.origin_d, r.created_at DESC""").fetchall(),
        columns=["origin_d", "model_version", "wape", "store_wape", "naive_wape", "bias"])
    first = served["model_version"].min()
    v1 = mlflow.lightgbm.load_model(f"models:/{M.MODEL_NAME}/{first}")

    frozen, naive_store = [], []
    for o in served["origin_d"]:
        rows = M.window(conn, o)
        f = rows[["id", "d"]].assign(forecast=M.predict(v1, rows))
        actual = M.actuals(conn, o + 1, o + 28)
        s = M.score(f, actual)
        frozen.append((s["wape"], s["store_wape"], s["bias"]))
        naive_store.append(M.score(M.seasonal_naive(conn, o), actual)["store_wape"])
    served[["frozen_wape", "frozen_store_wape", "frozen_bias"]] = frozen
    served["naive_store_wape"] = naive_store
    served["date"] = [feed.date_of(conn, o) for o in served["origin_d"]]
    served.to_csv("docs/results.csv", index=False, float_format="%.4f")

    events = pd.DataFrame(conn.execute("""SELECT c.date, e.event, e.model_version, e.reason FROM model_events e
                                          JOIN calendar c ON c.d = e.at_d ORDER BY e.at_d""").fetchall(),
                          columns=["date", "event", "version", "reason"])
    incident = conn.execute("""SELECT c.date, p_bad.wape, p_fix.wape FROM requests r1
        JOIN requests r2 ON r2.origin_d = r1.origin_d AND r2.created_at > r1.created_at
        JOIN performance p_bad ON p_bad.request_id = r1.request_id
        JOIN performance p_fix ON p_fix.request_id = r2.request_id
        JOIN calendar c ON c.d = r1.origin_d""").fetchall()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, col, title in [(axes[0], "", "Item-level WAPE"), (axes[1], "store_", "Store-total WAPE")]:
        ax.plot(served["date"], served[f"{col}wape"], color="#1f6feb", lw=2, label="served (monitored, retrained)")
        ax.plot(served["date"], served[f"frozen_{col}wape"], color="#d97706", lw=1.5, label=f"first model, never retrained (v{first})")
        ax.plot(served["date"], served["naive_wape" if not col else "naive_store_wape"], color="#9ca3af", lw=1,
                ls="--", label="seasonal naive")
        for _, e in events[events["event"] == "promoted"].iterrows():
            ax.axvline(e["date"], color="#16a34a", lw=0.8, alpha=0.6)
        for d, *_ in incident:
            ax.axvline(d, color="#dc2626", lw=1, ls=":")
        ax.set_title(title, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].text(0.01, 0.92, "green: new model promoted   red dotted: price feed incident",
                 transform=axes[1].transAxes, fontsize=8, color="#555")
    fig.suptitle("28-day forecasts for 3,049 items (Walmart CA_1), by forecast origin", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("docs/accuracy.png", dpi=130)

    summary = {
        "origins": len(served),
        "item_wape": {"served": served["wape"].mean(), "frozen": served["frozen_wape"].mean(),
                      "naive": served["naive_wape"].mean()},
        "store_wape": {"served": served["store_wape"].mean(), "frozen": served["frozen_store_wape"].mean(),
                       "naive": served["naive_store_wape"].mean()},
        "events": events.groupby("event").size().to_dict(),
        "incident": [{"origin": str(d), "wape_as_first_served": a, "wape_reissued": b} for d, a, b in incident],
    }
    print(json.dumps(summary, indent=2, default=float))
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
