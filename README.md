# Sales Forecasting Dashboard

Interactive dashboard that aggregates sales to monthly level, forecasts future sales with
three models (seasonal naive, Holt-Winters, SARIMA), and backtests them to pick the best one.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Uses a built-in synthetic dataset by default. To use your own data (e.g. Kaggle "Superstore"),
choose **Upload CSV** in the sidebar and map the date and sales columns.

## Features
- KPI cards: total sales, average monthly sales, year-over-year change, best month
- Forecast chart with 95% prediction interval
- Filters by category and region
- Backtest table (MAE, RMSE, MAPE) comparing all models against a seasonal-naive baseline
- Download forecast as CSV

## Key findings (fill in after running on your data)
1. Which model won the backtest, and by how much over the naive baseline?
2. Which months drive the most revenue?
3. Year-over-year growth and what it implies for planning.

## Next steps
- Add Prophet as a fourth model
- Add holiday/promo regressors (SARIMAX exog)
- Deploy on Streamlit Community Cloud
