SET client_min_messages = warning;
-- Source data, loaded by the feed as time passes (mlops/feed.py).
CREATE TABLE IF NOT EXISTS calendar (
    d            int PRIMARY KEY,
    date         date NOT NULL UNIQUE,
    wm_yr_wk     int NOT NULL,
    event_name_1 text,
    event_type_1 text,
    snap_ca      smallint NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id       text PRIMARY KEY,
    item_id  text NOT NULL,
    dept_id  text NOT NULL,
    cat_id   text NOT NULL,
    store_id text NOT NULL
);

-- Actual unit sales, arrive daily after the fact.
CREATE TABLE IF NOT EXISTS sales (
    id    text NOT NULL REFERENCES items,
    d     int  NOT NULL REFERENCES calendar,
    sales int  NOT NULL,
    PRIMARY KEY (id, d)
);
CREATE INDEX IF NOT EXISTS sales_d ON sales (d);

-- Shelf prices, published 28 days ahead (the price plan). NULL = not on sale.
CREATE TABLE IF NOT EXISTS prices (
    id         text NOT NULL REFERENCES items,
    d          int  NOT NULL REFERENCES calendar,
    sell_price real,
    PRIMARY KEY (id, d)
);
CREATE INDEX IF NOT EXISTS prices_d ON prices (d);

-- Average shelf price per department and day: a feature that depends on
-- every item in the department, so it is computed here and never from the
-- subset of items in one request.
CREATE OR REPLACE VIEW dept_prices AS
SELECT i.store_id, i.dept_id, p.d, avg(p.sell_price)::real AS dept_avg_price
FROM prices p JOIN items i USING (id)
GROUP BY 1, 2, 3;

CREATE TABLE IF NOT EXISTS feed_batches (
    batch_id  bigserial PRIMARY KEY,
    kind      text NOT NULL,          -- sales | prices
    d_from    int  NOT NULL,
    d_to      int  NOT NULL,
    rows      int  NOT NULL,
    note      text,
    loaded_at timestamptz NOT NULL DEFAULT now()
);

-- Everything the service answered.
CREATE TABLE IF NOT EXISTS requests (
    request_id    uuid PRIMARY KEY,
    model_version int  NOT NULL,
    origin_d      int  NOT NULL,
    n_items       int  NOT NULL,
    latency_ms    real NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS predictions (
    request_id    uuid NOT NULL,
    model_version int  NOT NULL,
    origin_d      int  NOT NULL,
    id            text NOT NULL,
    d             int  NOT NULL,
    forecast      real NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_origin ON predictions (origin_d);

-- The exact feature rows the model saw, for a fixed 10% sample of items.
CREATE TABLE IF NOT EXISTS prediction_features (
    request_id uuid  NOT NULL,
    origin_d   int   NOT NULL,
    id         text  NOT NULL,
    d          int   NOT NULL,
    features   jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS prediction_features_origin ON prediction_features (origin_d);

-- One row per check per monitoring run; long format so Grafana can plot it.
-- Rows are never deleted: re-running the monitor for a day (e.g. after a fix)
-- adds a new run, so the alert that triggered the fix stays on record.
CREATE TABLE IF NOT EXISTS monitor_results (
    run_d      int  NOT NULL,          -- business day the monitor ran for
    run_at     timestamptz NOT NULL,   -- when it ran
    check_name text NOT NULL,          -- data_quality | input_drift | performance | model_age
    subject    text NOT NULL,          -- feature, metric or rule
    value      real,
    threshold  real,
    status     text NOT NULL,          -- ok | warn | alert
    detail     text,
    PRIMARY KEY (run_d, run_at, check_name, subject)
);

-- Training, promotion and rejection decisions.
CREATE TABLE IF NOT EXISTS model_events (
    at_d          int  NOT NULL,
    event         text NOT NULL,       -- trained | promoted | rejected | skipped
    model_version int,
    reason        text,
    metrics       jsonb,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- Accuracy of each served forecast once its 28 days of actuals are in.
CREATE TABLE IF NOT EXISTS performance (
    request_id    uuid PRIMARY KEY,
    origin_d      int  NOT NULL,
    model_version int  NOT NULL,
    wape          real NOT NULL,
    store_wape    real NOT NULL,
    naive_wape    real NOT NULL,
    rel_wape      real NOT NULL,
    bias          real NOT NULL,
    expected_wape real
);
