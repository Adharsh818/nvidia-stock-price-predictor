# NVIDIA Stock Price Predictor

I built this project to learn how machine learning can be used to predict stock prices. It uses historical NVIDIA (NVDA) data from Alpaca to predict the next trading day's closing price.

## How it works

I downloaded daily stock data and created six features:
- Closing price
- Trading volume
- 5-day volatility
- 5-day moving average
- 10-day moving average
- 20-day moving average

I used the first 80% of the data for training and the remaining 20% for testing, keeping the dates in order.

## Models I used

- Random Forest
- XGBoost
- Linear Regression

I also added a simple baseline that predicts tomorrow's closing price will be the same as today's.

## Results

I compared the predictions using Mean Absolute Error (MAE). Lower values mean smaller errors.

| Model | MAE (USD) |
|---|---:|
| Baseline | 3.78 |
| Linear Regression | 4.12 |
| Random Forest | 11.05 |
| XGBoost | 15.69 |

The baseline performed better than all three models. Linear Regression looked close to the actual prices on the chart, but its average error was still higher than the baseline.

## Libraries used

Python, pandas, Matplotlib, scikit-learn, XGBoost, alpaca-py, and python-dotenv.

## Output files

- `model_comparison.png` — chart of actual and predicted prices
- `model_scores.csv` — MAE scores
- `next_session_forecasts.csv` — forecasts with the data cutoff date

## Prediction comparison

![Actual prices compared with model predictions](model_comparison.png)

## What I learned

A model can look accurate on a chart without beating a simple prediction method. Comparing it against a baseline helped me understand the results better.

The current dataset ends on November 13, 2026. These results are from one stock and one test period, so they don't show how the models would perform in every market condition.
