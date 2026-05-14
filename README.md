# ISA 444 — Walmart Weekly Sales Forecasting

Final project for ISA 444 (Business Forecasting). The goal is to forecast weekly sales for Walmart stores and departments using the Kaggle [Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) dataset, comparing nine forecasting models from the Nixtla ecosystem and the Amazon Chronos foundation model.

The project follows the same workflow taught in class with the Duke Energy load-forecasting notebooks, adapted to retail weekly sales.

---

## Project Highlights

- **20 top-volume Store/Department series** forecast at a **4-week monthly horizon**
- **5-fold non-overlapping time-series cross-validation**
- **9 models compared**: Naive, SeasonalNaive, AutoETS, AutoARIMA, AutoARIMA with exogenous regressors, LightGBM, AutoNBEATS, AutoNHITS, and Chronos
- **Exogenous features** (holidays, markdown promotions, store characteristics, weather) used where the model class supports them
- **Per-series, per-metric evaluation** (ME, MAE, RMSE, MAPE) with model win counts
- **Forecast-vs-actual plots** for every series
- Runs **entirely on a consumer laptop** (8 GB RAM, no GPU) with all heavy steps cached to disk

---

## Repository Layout

```
ISA444-Walmart-Project/
├── notebooks/
│   ├── 01_walmart_etl.ipynb              ← load + merge + select top 20 + feature engineering
│   ├── 02_walmart_statforecast.ipynb     ← Naive, SeasonalNaive, AutoETS, AutoARIMA (+ exog variant)
│   ├── 03_walmart_mlforecast.ipynb       ← LightGBM with lags + rolling stats + exog
│   ├── 04_walmart_neuralforecast.ipynb   ← AutoNBEATS + AutoNHITS (univariate)
│   ├── 05_walmart_chronos.ipynb          ← Chronos foundation model via TimeCoPilot
│   └── 06_walmart_evaluation.ipynb       ← synthesis: merge, win counts, plots, summary
├── outputs/
│   ├── cv_results/                       ← per-fold predictions and per-series evaluation
│   ├── forecasts/                        ← future forecasts for Kaggle test dates
│   ├── plots/                            ← aggregate charts + 20 forecast-vs-actual PNGs
│   └── summaries/                        ← win counts, headline metrics, final_summary.md
├── data/                                 ← (gitignored) processed parquet files written by notebook 01
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How To Reproduce

### 1. Download the raw Walmart data from Kaggle

Get `train.csv`, `features.csv`, `stores.csv`, and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data). Place them anywhere on your machine.

### 2. Update the path in `01_walmart_etl.ipynb`

Open Notebook 01 and edit the `RAW_DIR` constant near the top to point at the folder where you put the four CSVs:

```python
RAW_DIR = Path(r"C:\path\to\your\walmart-data")
```

### 3. Install the Python environment

```bash
pip install -r requirements.txt
```

Python 3.11 is recommended (what this project was developed on).

### 4. Run the notebooks in order

`01 → 02 → 03 → 04 → 05 → 06`

Each notebook writes its outputs to `../outputs/` so the next one can read them. Notebook 06 just combines everything.

### Approximate Runtimes (on an 8 GB laptop, no GPU)

| Notebook | Runtime |
|---|---|
| 01 — ETL | <1 minute |
| 02 — StatsForecast | 5–10 minutes |
| 03 — LightGBM | 1–2 minutes |
| **04 — NeuralForecast** | **30–60 minutes** (one-time; cached after) |
| 05 — Chronos | 5–15 minutes (one-time; downloads ~200MB on first run) |
| 06 — Synthesis | 2–3 minutes |

Notebook 04 caches its outputs aggressively. On re-runs it skips straight to evaluation — under 1 minute.

---

## Methodology

### Series selection

The top 20 Store/Department combinations by total weekly sales over the full Walmart history. No hand-picking — pure rank-by-volume. This biases toward grocery and consumables departments (which are high-volume and holiday-sensitive), giving the comparison a uniform retail context.

### Cross-validation

5 non-overlapping folds, step size 4 weeks, horizon 4 weeks. Same configuration across every model so metrics are directly comparable.

### Seasonality

`season_length = 52` everywhere it's exposed (SeasonalNaive, AutoETS, AutoARIMA). Walmart sales are driven by annual cycles (Easter, Memorial Day, Independence Day, Labor Day, Thanksgiving/Black Friday, Christmas). Even though some series have only ~143 weeks of history, capturing the full annual cycle was prioritized over fitting stability — StatsForecast falls back to non-seasonal models when there isn't enough data.

### Exogenous features

Used where the model class supports them:

| Model | Exog usage |
|---|---|
| Naive, SeasonalNaive, AutoETS, Chronos, AutoNBEATS, AutoNHITS | Univariate only |
| AutoARIMA_X | `is_holiday_week`, MarkDown1-5, Temperature |
| LightGBM | All the above plus lag features (1, 2, 4, 13, 26, 52), rolling means/stds, calendar features, and static store features (one-hot `store_type`, `store_size`) |

This split lets us compare what each model class extracts from the same data.

### Foundation model

Amazon's `chronos-bolt-small` (48M parameters) used in zero-shot mode through TimeCoPilot's `TimeCopilotForecaster`. No fine-tuning, no Walmart-specific training — the model sees each series' history once and outputs a forecast. ~200MB download on first use, cached locally afterward.

---

## Key Findings

See [`outputs/summaries/final_summary.md`](outputs/summaries/final_summary.md) for the auto-generated results breakdown. Brief observations:

- **Models with access to the holiday flag** (AutoARIMA_X, LightGBM) tend to dominate on Thanksgiving and Christmas weeks because they can predict the holiday spike.
- **SeasonalNaive is a stubbornly strong baseline** — it gets last year's holiday spike "for free" by copying the same-week-last-year value.
- **Univariate deep models** (AutoNBEATS, AutoNHITS) learn the general seasonal shape but **smooth over sharp holiday peaks** without exogenous features to anchor them.
- **Chronos** is competitive out of the box despite never having seen Walmart-specific data — a strong demonstration of the zero-shot foundation model paradigm.
- **MAPE was safe** to compute on all 20 series because minimum weekly sales is well above $1,000. On lower-volume departments MAPE would have been unreliable, and the writeup leans on MAE/RMSE in that case.

---

## Tools Used

All forecasting is restricted to the **Nixtlaverse** and **TimeCoPilot** ecosystems per project requirements:

- `statsforecast` — baseline and statistical models
- `mlforecast` + `lightgbm` — gradient-boosted tree forecasting
- `neuralforecast` — AutoNBEATS, AutoNHITS
- `timecopilot` — Chronos foundation model wrapper
- `utilsforecast` — uniform evaluation metrics and CV utilities
- `pandas`, `numpy`, `matplotlib` for data handling and plotting
- `optuna` as the hyperparameter search backend for the neural models

No `statsmodels`, no `prophet`, no cloud APIs, no GPU dependencies.

---

## Author

Michael Wardle
