# Step 1 of the data build: screen the JKP US stock panel and seal the 2021-2025 holdout.
# Run from the repo folder:  python data_prep/1_prep_panel.py
# Input: data/jkp_usa.parquet, a WRDS download of the JKP global factor data (USA), columns:
#   id permno gvkey eom excntry common obs_main primary_sec me prc ret_exc_lead1m crsp_shrcd
#   be_me gp_at qmj ret_12_1 niq_su at_gr1 ope_be   (eom 2003-01 .. 2025-12)
# Output: data/model_panel.parquet (eom <= 2020-11) and sealed/holdout_2021_2025.parquet (eom >= 2020-12).
import os
import pandas as pd

df = pd.read_parquet("data/jkp_usa.parquet")
df["eom"] = pd.to_datetime(df["eom"])
n0 = len(df)
df = df[(df["common"] == 1) & (df["primary_sec"] == 1) & (df["obs_main"] == 1)]
df = df[df["prc"] >= 5]                                                            # no penny stocks
df = df[df["me"] > df.groupby("eom")["me"].transform(lambda s: s.quantile(0.20))]  # drop the smallest 20% each month
df = df[df["ret_exc_lead1m"].notna()]
bad = df["ret_exc_lead1m"].abs() > 1.0                                             # corrupt JKP return records
print("rows raw", n0, "after screens", len(df), "| dropped |ret| > 100%:", int(bad.sum()))
df = df[~bad]
feats = ["be_me", "gp_at", "qmj", "ret_12_1", "niq_su", "at_gr1", "ope_be"]
out = df[["eom", "permno", "me"] + feats + ["ret_exc_lead1m"]].rename(columns={"ret_exc_lead1m": "ret_next"})
print("rows without permno dropped:", int(out["permno"].isna().sum()))
out = out[out["permno"].notna()].copy()
out["permno"] = out["permno"].astype("int64")
out[feats + ["me", "ret_next"]] = out[feats + ["me", "ret_next"]].astype("float64")
out = out.sort_values(["eom", "permno"]).reset_index(drop=True)
# ret_next at eom 2020-11 is the December 2020 return, so the model file ends there.
os.makedirs("sealed", exist_ok=True)
out[out["eom"] <= "2020-11-30"].to_parquet("data/model_panel.parquet", index=False)
out[out["eom"] > "2020-11-30"].to_parquet("sealed/holdout_2021_2025.parquet", index=False)
print("model file", int((out["eom"] <= "2020-11-30").sum()), "rows | holdout", int((out["eom"] > "2020-11-30").sum()), "rows")
