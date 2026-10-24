"""M5 zip -> Parquet for one store (CA_1), from 2014-01-01 to the end of the data.

    python -m mlops.prepare            # downloads data/raw/m5.zip if missing

The zip is the public Nixtla mirror of the M5 competition data (real Walmart
sales), checked against a SHA-256.
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
RAW = Path("data/raw/m5.zip")
OUT = Path("data")
STORE = "CA_1"
FIRST_D = 1069  # 2014-01-01 (d_1 = 2011-01-29)


def fetch() -> None:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{RAW}: unexpected SHA-256 {digest}")


def main() -> None:
    fetch()
    with zipfile.ZipFile(RAW) as z:
        cal = pd.read_csv(z.open("calendar.csv"), parse_dates=["date"])
        cal.insert(0, "d", np.arange(1, len(cal) + 1))
        cal = cal[cal["d"] >= FIRST_D]

        train = pd.read_csv(z.open("sales_train_evaluation.csv"))
        test = pd.read_csv(z.open("sales_test_evaluation.csv"))
        ids = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        wide = train.merge(test, on=ids, how="inner", validate="one_to_one")
        wide = wide[wide["store_id"] == STORE].copy()
        wide.insert(0, "id", wide["item_id"] + "_" + wide["store_id"])

        prices = pd.read_csv(z.open("sell_prices.csv"))
        prices = prices[prices["store_id"] == STORE]

    items = wide[["id", "item_id", "dept_id", "cat_id", "store_id"]].sort_values("id")
    day_cols = [f"d_{d}" for d in cal["d"]]
    sales = wide[["id", *day_cols]].melt(id_vars="id", var_name="d", value_name="sales")
    sales["d"] = sales["d"].str[2:].astype(np.int32)
    sales["sales"] = sales["sales"].astype(np.int32)

    # weekly prices -> one row per item and day (NULL where the item was not on sale)
    daily = cal[["d", "wm_yr_wk"]].merge(prices, on="wm_yr_wk")
    daily["id"] = daily["item_id"] + "_" + daily["store_id"]
    grid = sales[["id", "d"]].merge(daily[["id", "d", "sell_price"]], on=["id", "d"], how="left")

    calendar = cal.assign(snap_ca=cal["snap_CA"].astype(np.int16))[
        ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "snap_ca"]]

    OUT.mkdir(exist_ok=True)
    calendar.to_parquet(OUT / "calendar.parquet", index=False)
    items.to_parquet(OUT / "items.parquet", index=False)
    sales.sort_values(["id", "d"]).to_parquet(OUT / "sales.parquet", index=False)
    grid.sort_values(["id", "d"]).to_parquet(OUT / "prices.parquet", index=False)
    print(f"{len(items)} items, {len(calendar)} days "
          f"({calendar['date'].min():%Y-%m-%d} .. {calendar['date'].max():%Y-%m-%d}), "
          f"{len(sales):,} sales rows")


if __name__ == "__main__":
    main()
