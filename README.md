# Does option skew predict stock returns? Pre- vs post-2016

**Result:** a stock's option skew predicted its return relative to other stocks over the next month from 2003 to 2015 (IC -0.018, t -4.4 for 2003-2012), then stopped working after 2016 (IC -0.002, t -0.27). Inside a model it adds nothing to 8 known factors on 2016-2020. The test was pre-registered and the signal is **killed on its decay rule**.

## Hypothesis

When one stock's puts get expensive compared with other stocks' puts, someone with bad news is buying downside with leverage. Index funds, long-only funds and most retail holders do not read option skew and often cannot short, so the information reaches the stock price slowly. Index-wide skew is a different story (fear, the price of insurance), which is why skew is ranked within each month: the market-wide part cancels out.

## Pre-registration (written before any result)

| | |
|---|---|
| feature | skew = 30-day put implied vol (delta -25) minus call implied vol (delta 50), at each month end, ranked within the month |
| sign | negative: steeper skew, the stock lags its peers next month |
| pass | t >= 2 on the monthly IC, AND it must add to the 8 factors in a paired month-by-month test |
| decay | works 2003-2012 but not 2016-2020 = fail for trading |
| size | large and small stocks reported separately |

## Data

- **Stocks:** JKP global factor data (USA) via WRDS, 2003-2020. Common shares, primary listing, price >= $5, larger than the smallest 20% each month; 326 corrupt return records (beyond +-100% in a month) removed. 608,633 stock-months, 215 months, about 2,700 stocks a month.
- **Target:** the stock's excess return over the next month, ranked within the month. The 8 factors: value (book/price), gross profitability, quality, momentum (12-1), earnings surprise, asset growth, operating profitability, size.
- **Skew:** OptionMetrics 30-day standardized volatility surface, linked to CRSP. 78% of stock-months have it (57% in 2003, about 90% from 2012; 47% of the smallest fifth of stocks, 98% of the largest).
- **Data trap found:** OptionMetrics has rows with empty implied vols from 2020-07-27 into August 2020, so July 2020 uses the 2020-07-24 surface.
- **2021-2025 is held out and has not been opened.**

## Method

- IC = Spearman rank correlation between a signal and next-month returns, computed once per month.
- **The t-stat counts months, not stock-months.** All stocks in a month share that month's news, so one month is one bet. Counting rows would turn the linear model's t of 0.68 into a fake 3.29.
- Model test: train 2003-2012, early-stopping slice 2013-2015, validation 2016-2020. Linear regression and XGBoost (max depth 3, learning rate 0.05, early stopping), each with the 8 factors and with 8 factors + skew, identical settings.

## Results

Skew on its own (no fitting):

| period | mean IC | t | months | share of months > 0 |
|---|---|---|---|---|
| 2003-2020 | -0.0153 | -4.52 | 215 | 0.38 |
| 2003-2012 | -0.0180 | -4.40 | 120 | 0.33 |
| 2013-2015 | -0.0286 | -2.74 | 36 | 0.36 |
| 2016-2020 | -0.0017 | -0.27 | 59 | 0.49 |

By size (above / below the month's median market cap):

| | 2003-2020 IC (t) | 2016-2020 IC (t) |
|---|---|---|
| large | -0.0098 (-2.33) | +0.0043 (0.56) |
| small | -0.0153 (-5.18) | -0.0047 (-0.89) |

Inside the model, validation 2016-2020:

| | 8 factors | 8 factors + skew | paired difference, t |
|---|---|---|---|
| linear | 0.0084 (t 0.68) | 0.0087 (t 0.73) | 0.20 |
| XGBoost | 0.0113 (t 0.86) | 0.0101 (t 0.78) | -0.55 |

## Reading

- The sign was right and the effect was strong for 13 years, stronger in small stocks, which fits a slow-holder story.
- It still worked in 2013-2015, after the original paper (Xing, Zhang and Zhao, 2010, JFQA), and died around 2016 in both large and small stocks. That fits more systematic trading of option data after 2016, but 59 months is one regime: it cannot prove why. Published predictors lose a large part of their returns after publication on average (McLean and Pontiff, 2016, JF).
- XGBoost gave skew about 11% of its feature importance and kept more trees (249 vs 107): it learned the 2003-2012 pattern, which did not pay after 2016. Importance shows what a model used, not what works out of sample.
- The 8 published factors themselves show no provable edge on 2016-2020 either.

## Reproduce

Needs a WRDS account with JKP, OptionMetrics and CRSP. Data cannot be redistributed, so none is included.

```
pip install -r requirements.txt
# put the JKP USA download at data/jkp_usa.parquet (columns listed in data_prep/1_prep_panel.py)
set WRDS_USERNAME=your_username
python data_prep/1_prep_panel.py
python data_prep/2_pull_skew.py
python data_prep/3_add_skew.py
python skew_test.py
```

## Credits

Research question, hypothesis, pre-registration and analysis code: Matthijs Kok. The data-pull scripts in `data_prep/` were written with AI assistance.
