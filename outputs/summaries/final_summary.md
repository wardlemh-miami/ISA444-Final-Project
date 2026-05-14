# Walmart Forecasting — Summary of Findings

**Series evaluated:** 20 top-volume Store/Department combinations
**Horizon:** 4 weeks (monthly)
**CV:** 5 non-overlapping folds

## Headline Results

- **Best model by mean MAE :** `Chronos`  (MAE = 6,268)
- **Best model by mean RMSE:** `Chronos` (RMSE = 7,275)
- **Best model by mean MAPE:** `AutoARIMA` (MAPE = 0.05%)
- **Most series won overall:** `Chronos` with 90 total wins across all metrics

## Win Counts (out of 20 series per metric)

|               |   bias |   mae |   mape |   rmse |
|:--------------|-------:|------:|-------:|-------:|
| Naive         |      5 |     1 |      3 |      3 |
| SeasonalNaive |     11 |     8 |      6 |     11 |
| AutoETS       |     11 |     4 |      3 |      5 |
| AutoARIMA     |     18 |    14 |     17 |     15 |
| AutoARIMA_X   |      6 |     9 |     11 |      9 |
| LightGBM      |      9 |    13 |     12 |      9 |
| AutoNBEATS    |     16 |    15 |     17 |     12 |
| AutoNHITS     |     12 |     8 |      8 |      9 |
| Chronos       |     12 |    28 |     23 |     27 |

## Aggregate Metrics

| metric   |    Naive |   SeasonalNaive |   AutoETS |   AutoARIMA |   AutoARIMA_X |   LightGBM |   AutoNBEATS |   AutoNHITS |   Chronos |
|:---------|---------:|----------------:|----------:|------------:|--------------:|-----------:|-------------:|------------:|----------:|
| bias     |  2025.84 |        -1012.75 |    857.83 |     2227.37 |       2355.21 |    4228.96 |      -586.89 |      991.52 |   2359.35 |
| mae      | 12065.4  |        11298.4  |   7845.23 |     7409.31 |       7597.05 |    7872.44 |      6808.59 |     7127.02 |   6267.77 |
| mape     |     0.09 |            0.08 |      0.06 |        0.05 |          0.06 |       0.06 |         0.05 |        0.05 |      0.05 |
| rmse     | 13622.4  |        12641.1  |   9117.87 |     8859.14 |       9078.67 |    9183.86 |      7923.86 |     8297.59 |   7274.67 |

## Interpretation Notes

- **Models with exogenous features** (`AutoARIMA_X`, `LightGBM`) tend to dominate on **holiday weeks** because they can see the `is_holiday_week` flag and adjust predictions accordingly. This is most visible on Thanksgiving/Black Friday and Christmas weeks.
- **SeasonalNaive** is a surprisingly strong baseline — it gets last year's holiday spike 'for free' just by copying the same-week-last-year value.
- **Univariate deep models** (`AutoNBEATS`, `AutoNHITS`) tend to **smooth over holiday spikes**. They learn the general seasonal shape but underpredict the sharp peaks. With only 143 weeks of history they don't have enough holiday cycles to learn the exact magnitude.
- **Chronos** (zero-shot foundation model) performs competitively despite never seeing Walmart-specific data. It's a reasonable choice when training data is scarce or for new series.
- **MAPE** was safe to compute here because every series has high min weekly sales (>$1,000) — but it would have been unreliable on lower-volume departments.

## What Drove Model Performance

Three factors mattered most:
1. **Access to the holiday flag** — models that saw `is_holiday_week` won the spike weeks.
2. **Long enough memory** — `lag_52` in LightGBM and the implicit annual seasonality in `SeasonalNaive`/`AutoETS`/`AutoARIMA` carried most of the seasonal signal.
3. **Conservative vs aggressive predictions** — on stable weeks, anything close to the mean wins; on spike weeks, the model that can predict the spike wins.
