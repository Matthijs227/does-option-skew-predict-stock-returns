# Step 3 of the data build: add skew to the model panel and to the sealed holdout.
# Run from the repo folder:  python data_prep/3_add_skew.py
import pandas as pd

sk = pd.read_parquet("data/skew_monthly.parquet")[["eom", "permno", "skew", "iv_put25", "iv_call50"]]
mp = pd.read_parquet("data/model_panel.parquet").merge(sk, on=["eom", "permno"], how="left")
hp = pd.read_parquet("sealed/holdout_2021_2025.parquet").merge(sk, on=["eom", "permno"], how="left")
mp.to_parquet("data/model_panel_v2.parquet", index=False)
hp.to_parquet("sealed/holdout_2021_2025_v2.parquet", index=False)
print("model_panel_v2", mp.shape, "skew coverage", round(mp["skew"].notna().mean(), 3))
print("holdout_v2", hp.shape, "skew coverage", round(hp["skew"].notna().mean(), 3))
