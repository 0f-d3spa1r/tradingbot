# tests/test_data_loader.py
from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET
from data_loader import set_client, get_processed_ohlcv

def main():
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)
    df = get_processed_ohlcv("BTCUSDT", "15", limit=200)
    print("OHLCV shape:", df.shape)
    print(df.tail(2))

if __name__ == "__main__":
    main()
