import logging
import pandas as pd
import pytest
from pybit.unified_trading import HTTP
from data_loader import set_client, fetch_ohlcv
from config import BYBIT_API_KEY, BYBIT_API_SECRET

logging.basicConfig(level=logging.INFO)

@pytest.mark.smoke
def test_fetch_ohlcv_basic():
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    df = fetch_ohlcv("BTCUSDT", "15", limit=200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 100, "Слишком короткая история"
    for col in ["open","high","low","close","volume"]:
        assert col in df.columns
    assert (df["close"] > 0).all()
