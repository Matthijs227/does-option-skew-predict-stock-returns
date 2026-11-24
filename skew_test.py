# Does a stock's option skew predict its return relative to other stocks next month,
# and does it add anything to 8 known factors? Split 2003-2012 vs 2016-2020.
#
# Hypothesis (written before any result): when ONE stock's puts get expensive compared with
# its peers, someone with bad news is buying downside with leverage. Index funds, long-only
# funds and most retail holders do not read option skew and often cannot short, so the news
# reaches the stock price slowly.
#
# Pre-registration:
#   feature  skew = 30-day put implied vol (delta -25) minus call implied vol (delta 50),
#            at each month end, ranked within the month
#   sign     negative: steeper skew -> the stock lags its peers next month
#   pass     t >= 2 on the monthly IC (t over months, not stock-months), AND it must add to the
#            8 factors in a paired month-by-month test
#   decay    works 2003-2012 but not 2016-2020 -> fail for trading
#   size     large and small stocks reported separately
#
# Data: data/model_panel_v2.parquet, built by the scripts in data_prep/ (see README).
#   One row = one US stock at one month end (eom), 2003-01 to 2020-11, 608,633 rows.
#   ret_next = the stock's excess return over the NEXT month. 2021-2025 is held out, unopened.
#   No options data -> skew is NaN (XGBoost handles it, the linear model fills 0 = the middle rank).

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

panel = pd.read_parquet("data/model_panel_v2.parquet")

# ---------------------------------------------------------------
# Step 1: target and features, all ranked within each month (-0.5 worst .. +0.5 best)
# Ranking removes the market move from the target and keeps feature levels comparable over time.
# ---------------------------------------------------------------
panel["size"]= np.log(panel["me"])
panel["large"]= panel["me"]>panel.groupby("eom")["me"].transform("median")
panel["small"]=panel["me"]<panel.groupby("eom")["me"].transform("median")
features= ["be_me","gp_at","qmj","ret_12_1","niq_su","at_gr1","ope_be","size"]
panel["target"]= panel.groupby("eom")["ret_next"].rank(pct=True)-0.5
for f in features + ["skew"]:
    panel[f]=panel.groupby("eom")[f].rank(pct=True)-0.5
print("share of stock-months with skew:", round(panel["skew"].notna().mean(), 3))


# ---------------------------------------------------------------
# Scoring: IC = Spearman correlation between a signal and next-month returns, one per month.
# The t-stat counts MONTHS: stocks in the same month share that month's news, so one month is one bet.
# ---------------------------------------------------------------
def ic_by_month(df,pred_col):
    out= {}
    for month, group in df.groupby("eom"):
        out[month] = group[pred_col].corr(group["ret_next"], method="spearman")
    return pd.Series(out)
def report(name, ic):
    t = ic.mean() / (ic.std() / len(ic) ** 0.5)
    print(f"{name:12s} mean IC {ic.mean():.4f}   sd {ic.std():.4f}   t {t:.2f}   months {len(ic)}   share of months > 0: {(ic > 0).mean():.2f}")


# ---------------------------------------------------------------
# Step 2: skew on its own (a raw signal, no fitting): full sample, then 2003-2012 vs 2016-2020
# ---------------------------------------------------------------
print("\nStep 2: skew alone")
has_skew= panel[panel["skew"].notna()]
skew_early= has_skew[has_skew["eom"]<"2013-01-01"]
skew_late= has_skew[has_skew["eom"]>="2016-01-01"]
report("skew",ic_by_month(has_skew,"skew"))
report("skew_early",ic_by_month(skew_early,"skew"))
report("skew_late",ic_by_month(skew_late,"skew"))
skew_mid= has_skew[(has_skew["eom"]>="2013-01-01")&(has_skew["eom"]<"2016-01-01")]   # the 3 years in between
report("skew_mid",ic_by_month(skew_mid,"skew"))


# ---------------------------------------------------------------
# Step 3: large vs small stocks (above / below that month's median market cap), and each after 2016
# ---------------------------------------------------------------
print("\nStep 3: by size")
skew_large= has_skew[has_skew["large"]]
skew_small= has_skew[has_skew["small"]]
report("skew_large",ic_by_month(skew_large,"skew"))
report("skew_small",ic_by_month(skew_small,"skew"))
large_late= has_skew[(has_skew["large"])&(has_skew["eom"]>="2016-01-01")]
small_late= has_skew[(has_skew["small"])&(has_skew["eom"]>="2016-01-01")]
report("large_late", ic_by_month(large_late,"skew"))
report("small_late", ic_by_month(small_late,"skew"))


# ---------------------------------------------------------------
# Step 4: does skew ADD to the 8 factors? Same split and settings for both versions:
#   train 2003-2012, stop (early-stopping slice) 2013-2015, valid 2016-2020.
# The pass is the PAIRED test: each month's IC with skew minus without skew, t over the 59 months.
# ---------------------------------------------------------------
print("\nStep 4: 8 factors vs 8 factors + skew, scored on 2016-2020")
features9= features+["skew"]
train = panel[panel["eom"] < "2013-01-01"].copy()
stop = panel[panel["eom"].between("2013-01-01","2016-01-01")].copy()
valid = panel[panel["eom"]>"2016-01-01"].copy()

lin8= LinearRegression()
lin8.fit(train[features].fillna(0),train["target"])
valid["pred_lin8"]= lin8.predict(valid[features].fillna(0))
report("linear 8", ic_by_month(valid, "pred_lin8"))

lin= LinearRegression()
lin.fit(train[features9].fillna(0),train["target"])
valid["pred_lin9"]= lin.predict(valid[features9].fillna(0))
print("linear 9 coefficients:", dict(zip(features9, lin.coef_.round(4))))
report("linear 9", ic_by_month(valid, "pred_lin9"))

model8= xgb.XGBRegressor(n_estimators=2000,max_depth=3, learning_rate=0.05, early_stopping_rounds=50,random_state=0)
model8.fit(train[features],train["target"], eval_set=[(stop[features],stop["target"])],verbose=False)
valid["pred_xgb8"] = model8.predict(valid[features])
print("xgboost 8 trees kept:", model8.best_iteration)
report("xgboost 8", ic_by_month(valid, "pred_xgb8"))

model= xgb.XGBRegressor(n_estimators=2000,max_depth=3, learning_rate=0.05, early_stopping_rounds=50,random_state=0)
model.fit(train[features9],train["target"], eval_set=[(stop[features9],stop["target"])],verbose=False)
print("xgboost 9 trees kept:", model.best_iteration)
valid["pred_xgb9"] = model.predict(valid[features9])
report("xgboost 9", ic_by_month(valid, "pred_xgb9"))
print("xgboost 9 importance:", dict(zip(features9, model.feature_importances_.round(3))))

report("lin 9 - 8", ic_by_month(valid, "pred_lin9") - ic_by_month(valid, "pred_lin8"))
report("xgb 9 - 8", ic_by_month(valid, "pred_xgb9") - ic_by_month(valid, "pred_xgb8"))
