# ISA 444 Walmart Retail Sales Forecasting

## Project Overview
We forecast weekly Walmart sales by store and department using time-series cross-validation and compare baseline, statistical, machine learning, neural, and foundation models.

## Forecasting Task
Dataset: Walmart Kaggle Forecasting Challenge  
Target: Weekly sales  
Series: Selected store-department combinations  
Horizon: weekly forecasts

## Models Compared
- Naive
- Seasonal Naive
- AutoETS
- AutoARIMA
- LightGBM via MLForecast
- AutoNBEATS
- AutoNHITS
- Chronos

## Evaluation
Metrics:
- ME
- MAE
- RMSE
- MAPE

## Results
See `outputs/` for CSV results and `plots/` for forecast-vs-actual charts.

## Main Findings
Write 1–2 paragraphs explaining which model performed best overall and why.

## Reproducibility
Run the notebooks in order from `01` to `06`.
