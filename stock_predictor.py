import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime, timezone

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

import os
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# Find .env beside this Python file.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# Read the credentials without displaying them.
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    raise ValueError("Add both Alpaca keys to your .env file.")

client = StockHistoricalDataClient(api_key, secret_key)

print("Credentials loaded. Data client created.")

# Download NVIDIA's historical daily prices.
SYMBOL = "NVDA"

request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=datetime(2022, 1, 1, tzinfo=timezone.utc),
    end=datetime(2026, 11, 14, tzinfo=timezone.utc),
    feed=DataFeed.IEX,
    adjustment=Adjustment.SPLIT,
)

bars = client.get_stock_bars(request)
data = bars.df

if data.empty:
    raise ValueError("No stock data returned. Check your request.")

# Select NVIDIA and put the trading dates in order.
df = data.xs(SYMBOL, level="symbol").sort_index()

print(df.head())
print(f"\nTotal trading days: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# FEATURES: information available after each trading day's close.
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(window=5).std()

df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_10"] = df["close"].rolling(window=10).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()

# TARGET: the next trading day's closing price.
df["target"] = df["close"].shift(-1)

features = [
    "close",
    "volume",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_20",
]

# Keep the newest features for a later future prediction.
latest_features = df[features].iloc[[-1]].copy()

# Remove rows without enough history or a known target.
model_data = df.dropna(subset=features + ["target"]).copy()

print("\nPrepared data:")
print(model_data[features + ["target"]].head())
print(f"\nRows available for modelling: {len(model_data)}")

# Model inputs and expected answers.
X = model_data[features]
y = model_data["target"]

# Split chronologically: older data first, newer data last.
split = int(len(model_data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

print(
    f"Training feature dates: "
    f"{X_train.index.min()} to {X_train.index.max()}"
)
print(
    f"Testing feature dates: "
    f"{X_test.index.min()} to {X_test.index.max()}"
)
# Train the Random Forest model.
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
)

rf.fit(X_train, y_train)

# Predict next-day closing prices for the test set.
rf_preds = rf.predict(X_test)

# Measure the average absolute prediction error.
rf_mae = mean_absolute_error(y_test, rf_preds)

print(f"\nRandom Forest MAE: ${rf_mae:.2f}")
# Baseline: predict that the next close equals today's close.
baseline_preds = X_test["close"]
baseline_mae = mean_absolute_error(y_test, baseline_preds)

print(f"Baseline MAE: ${baseline_mae:.2f}")
# Train XGBoost using the settings shown in the video.
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

xgb.fit(X_train, y_train)

# Predict and evaluate on the same test set.
xgb_preds = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)

print(f"\nBaseline MAE:      ${baseline_mae:.2f}")
print(f"Random Forest MAE: ${rf_mae:.2f}")
print(f"XGBoost MAE:       ${xgb_mae:.2f}")
# Train and evaluate Linear Regression.
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)

print(f"\nLinear Regression MAE: ${lr_mae:.2f}")

# Compare all models against the baseline.
scores = pd.Series({
    "Baseline": baseline_mae,
    "Random Forest": rf_mae,
    "XGBoost": xgb_mae,
    "Linear Regression": lr_mae,
}).sort_values()

print("\nMAE comparison — lower is better:")
print(scores.round(2))

scores.rename("MAE_USD").to_csv("model_scores.csv")

# Refit fresh models using all rows with known targets.
forecast_models = {
    "Random Forest": clone(rf),
    "XGBoost": clone(xgb),
    "Linear Regression": clone(lr),
}

latest_close = float(latest_features["close"].iloc[0])

# Baseline forecast: the next close equals the latest close.
forecasts = {"Baseline": latest_close}

for name, model in forecast_models.items():
    model.fit(X, y)
    forecasts[name] = float(model.predict(latest_features)[0])

forecast_table = pd.DataFrame.from_dict(
    forecasts,
    orient="index",
    columns=["Predicted close"],
)

forecast_table["Change (%)"] = (
    forecast_table["Predicted close"] / latest_close - 1
) * 100

print(f"\nLatest data date: {latest_features.index[0].date()}")
print(f"Latest closing price: ${latest_close:.2f}")
print("\nForecast for the next trading session:")
print(forecast_table.round(2))

forecast_export = forecast_table.copy()

forecast_export.insert(
    0, "Data cutoff", latest_features.index[0].date()
)
forecast_export.insert(
    1, "Latest close", latest_close
)
forecast_export["Forecast horizon"] = "Next trading session"

forecast_export.to_csv(
    "next_session_forecasts.csv",
    index_label="Model",
)
# Each prediction targets the trading day AFTER its feature date.
target_dates = pd.Series(
    df.index,
    index=df.index,
).shift(-1)

test_dates = pd.DatetimeIndex(
    target_dates.loc[X_test.index]
)

plt.figure(figsize=(14, 7))

plt.plot(
    test_dates,
    y_test.to_numpy(),
    label="Actual",
    color="black",
    linewidth=2,
)

plt.plot(
    test_dates,
    rf_preds,
    label="Random Forest",
    alpha=0.8,
)

plt.plot(
    test_dates,
    xgb_preds,
    label="XGBoost",
    alpha=0.8,
)

plt.plot(
    test_dates,
    lr_preds,
    label="Linear Regression",
    alpha=0.8,
)

plt.plot(
    test_dates,
    baseline_preds.to_numpy(),
    label="Baseline: previous close",
    color="purple",
    linestyle="--",
    alpha=0.7,
)

plt.title("NVDA Price Prediction Model Comparison")
plt.xlabel("Target trading date")
plt.ylabel("Split-adjusted price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=150)
plt.show()