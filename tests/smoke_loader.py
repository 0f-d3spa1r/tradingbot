import logging
from pybit.unified_trading import HTTP
from data_loader import set_client, fetch_ohlcv
from config import get_settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    s = get_settings()
    client = HTTP()  # публичные свечи доступны без ключей
    set_client(client)

    df = fetch_ohlcv("BTCUSDT", s.interval, limit=500)
    print("Rows:", len(df))
    print("Index tz:", df.index.tz)
    print("First ts:", df.index[0])
    print("Last  ts:", df.index[-1])
    print(df.tail(3).to_string())
