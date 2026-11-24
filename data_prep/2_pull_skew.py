# Step 2 of the data build: 30-day implied vol skew per stock at each month end, OptionMetrics (WRDS).
# Run from the repo folder:  python data_prep/2_pull_skew.py
#   skew = IV(put, delta -25) - IV(call, delta 50), both from the 30-day standardized surface (optionm.vsurfdYYYY).
# Month end = the last weekday of the month with FILLED implied vols. Walking back matters: holidays have no
# surface, and OptionMetrics has rows with EMPTY implied vols from 2020-07-27 into August 2020 (July 2020 -> 07-24).
# Linked to CRSP permno with wrdsapps.opcrsphist (score 1, date inside the link window).
# Needs a WRDS account: set the environment variable WRDS_USERNAME. Output: data/skew_monthly.parquet.
import os, time
import pandas as pd
import wrds

t0 = time.time()
conn = {"db": None}
def fetch(q):
    for attempt in range(4):                       # WRDS drops long connections: reconnect and retry
        try:
            if conn["db"] is None:
                conn["db"] = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
            return conn["db"].raw_sql(q)
        except Exception as e:
            print("  retry", attempt + 1, type(e).__name__, flush=True)
            try:
                conn["db"].close()
            except Exception:
                pass
            conn["db"] = None
            time.sleep(20)
    raise RuntimeError("query failed 4 times")

os.makedirs("data/skew_raw", exist_ok=True)
parts = []
for yr in range(2003, 2026):
    path = f"data/skew_raw/{yr}.parquet"           # one cached file per year: a rerun resumes where it stopped
    if os.path.exists(path):
        parts.append(pd.read_parquet(path))
        continue
    chunks = []
    for me in pd.date_range(f"{yr}-01-31", f"{yr}-12-31", freq="ME"):
        for d in pd.bdate_range(me - pd.Timedelta(days=9), me)[::-1][:7]:
            x = fetch(f"""select secid, date, cp_flag, impl_volatility from optionm.vsurfd{yr}
                          where date = '{d:%Y-%m-%d}' and days = 30
                          and ((delta = -25 and cp_flag = 'P') or (delta = 50 and cp_flag = 'C'))""")
            x = x[x["impl_volatility"].notna()]
            if len(x):
                chunks.append(x)
                break
    x = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["secid", "date", "cp_flag", "impl_volatility"])
    x["date"] = pd.to_datetime(x["date"])
    x.to_parquet(path, index=False)
    parts.append(x)
    print(yr, "months found", x["date"].nunique(), "rows", len(x), round(time.time() - t0), "s", flush=True)

iv = pd.concat([p for p in parts if len(p)], ignore_index=True)
link = fetch("select secid, sdate, edate, permno from wrdsapps.opcrsphist where score = 1")
conn["db"].close()
link["sdate"] = pd.to_datetime(link["sdate"])
link["edate"] = pd.to_datetime(link["edate"])
w = iv.pivot_table(index=["secid", "date"], columns="cp_flag", values="impl_volatility").reset_index()
w = w.rename(columns={"P": "iv_put25", "C": "iv_call50"})
w["skew"] = w["iv_put25"] - w["iv_call50"]
m = w.merge(link, on="secid", how="inner")
m = m[(m["date"] >= m["sdate"]) & (m["date"] <= m["edate"])]
m["eom"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
m["permno"] = m["permno"].astype("int64")
m = m.sort_values(["permno", "eom", "secid"]).drop_duplicates(["permno", "eom"])
out = m[["eom", "permno", "secid", "iv_put25", "iv_call50", "skew"]].sort_values(["eom", "permno"])
out.to_parquet("data/skew_monthly.parquet", index=False)
print("skew rows", len(out), "months", out["eom"].nunique(), out["eom"].min().date(), "->", out["eom"].max().date(),
      "|", round(time.time() - t0), "s")
