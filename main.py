import matplotlib
matplotlib.use("TkAgg")  # Установить backend для корректного отображения графика
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import xgboost as xgb
from binance.client import Client
from datetime import datetime, timedelta
import time

# Binance API
api_key = "HKQsMm9CiiKqDHcHujvGJ9kLMFGRptJlIDeVq886bv4KcHIn0zyJTC3hNErTsZ0e"
api_secret = "Hcuhot76tkbQHn3fv4g7Wtelu8mmGt7ebWh6Z6DvlBu1mE32mzzMoLXWATXaFcJV"
client = Client(api_key, api_secret)


# === ФУНКЦИИ ДЛЯ БЭКТЕСТИНГА ===

# Получение исторических данных
def get_historical_data(symbol, interval, lookback_days=90):  # Уменьшен lookback
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)

    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%Y-%m-%d"),
                                          end_time.strftime("%Y-%m-%d"))
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                       "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                       "taker_buy_quote_asset_volume", "ignore"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df


# Получение топовых пар по объёму
def get_top_pairs():
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x["quoteVolume"]), reverse=True)
    top_pairs = [t["symbol"] for t in sorted_tickers if "USDC" in t["symbol"]][:5]  # Берём топ-5 пар с USDC
    return top_pairs


# Добавление индикаторов
def add_indicators(df):
    df["sma"] = ta.trend.sma_indicator(df["close"], window=14)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df.dropna(inplace=True)
    return df


# Загрузка XGBoost модели
model = xgb.Booster()
model.load_model("xgboost_model.json")


# Функция прогнозирования (исправленный вывод)
def predict_with_xgboost(df):
    features = df[["close", "sma", "rsi"]].values
    print(f"Форма данных перед XGBoost: {features.shape}")  # Проверяем форму данных
    dmatrix = xgb.DMatrix(features)
    predictions = model.predict(dmatrix)
    print(f"Прогнозы XGBoost (первые 10 значений): {predictions[:10]}")  # Проверяем 10 значений
    return predictions


# Бэктестинг стратегии (логирование + оптимизация)
def backtest(symbol, initial_balance=1000, risk_per_trade=0.02):
    df = get_historical_data(symbol, Client.KLINE_INTERVAL_1MINUTE)
    df = add_indicators(df)

    print(f"Бэктестинг {symbol}...")
    print(f"Длина DataFrame после удаления NaN: {len(df)}")
    print(f"Первые 10 строк данных перед подачей в XGBoost:\n{df[['close', 'sma', 'rsi']].head(10)}")

    # Проверка масштабов данных
    print("Минимумы тренировочных данных:", df[["close", "sma", "rsi"]].min())
    print("Максимумы тренировочных данных:", df[["close", "sma", "rsi"]].max())

    balance = initial_balance
    position = 0
    entry_price = 0
    trade_log = []

    predictions = predict_with_xgboost(df)  # Получаем все предсказания разом

    for i in range(len(df) - 1):
        if i % 100 == 0:  # Логируем прогресс каждые 100 итераций
            print(f"{symbol}: Обработано {i}/{len(df) - 1} свечей")

        if predictions[i] == 1 and position == 0:  # Покупка
            position = balance * risk_per_trade / df["close"].iloc[i]
            entry_price = df["close"].iloc[i]
            balance -= position * entry_price * 1.001  # Учитываем комиссию
            trade_log.append(("BUY", df["timestamp"].iloc[i], entry_price))

        elif predictions[i] == 0 and position > 0:  # Продажа
            balance += position * df["close"].iloc[i] * 0.999  # Учитываем комиссию
            trade_log.append(("SELL", df["timestamp"].iloc[i], df["close"].iloc[i]))
            position = 0

    final_balance = balance + (position * df["close"].iloc[-1]) if position > 0 else balance
    return final_balance, trade_log

# Запуск бэктеста
top_pairs = get_top_pairs()
results = {}

for pair in top_pairs:
    final_balance, trades = backtest(pair)
    results[pair] = final_balance
    print(f"Итоговый баланс: {final_balance:.2f} USDC")

# График сделок
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color="green")
plt.xlabel("Торговая пара")
plt.ylabel("Итоговый баланс (USDC)")
plt.title("Результаты бэктестинга")
plt.show()
plt.savefig("backtest_results.png")
