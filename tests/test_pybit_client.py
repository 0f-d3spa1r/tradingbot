# tests/test_pybit_client.py
from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET

def main():
    http = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    resp = http.get_kline(category="spot", symbol="BTCUSDT", interval="15", limit=5)
    ok = isinstance(resp, dict) and "result" in resp and "list" in resp["result"] and len(resp["result"]["list"]) > 0
    print("Pybit kline OK:", ok)
    if not ok:
        print("Raw response:", resp)

if __name__ == "__main__":
    main()
