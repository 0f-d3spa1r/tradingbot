# tests/smoke_env.py
import pytest
from config import BYBIT_API_KEY, BYBIT_API_SECRET  # config.py сам делает load_dotenv()

@pytest.mark.smoke
def test_env_vars_present():
    assert BYBIT_API_KEY, "BYBIT_API_KEY отсутствует (проверь .env в корне проекта)"
    assert BYBIT_API_SECRET, "BYBIT_API_SECRET отсутствует (проверь .env в корне проекта)"
