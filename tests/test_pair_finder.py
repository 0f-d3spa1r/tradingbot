import logging
import pytest
from pybit.unified_trading import HTTP

# ВАЖНО: подключаем оба сеттера — из data_loader и из pair_finder
from data_loader import set_client as set_dl_client
from pair_finder import get_candidate_pairs, set_client as set_pf_client
from config import BYBIT_API_KEY, BYBIT_API_SECRET, QUOTE_ASSETS

logging.basicConfig(level=logging.INFO)

@pytest.mark.smoke
def test_pair_finder_smoke():
    """
    Смоук-тест: инициализируем Pybit-клиент, пробуем получить кандидатов,
    проверяем базовые инварианты. Если биржа ничего не отдала (rate-limit / сеть),
    аккуратно скипаем тест.
    """
    # Один и тот же HTTP-клиент прокидываем В ОБА модуля
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_dl_client(client)
    set_pf_client(client)

    pairs = get_candidate_pairs(force_refresh=True)

    if not pairs:
        pytest.skip("No pairs returned (empty list). Possibly API rate-limited / network issue.")

    # Базовые проверки
    assert isinstance(pairs, list)
    assert all(isinstance(p, str) for p in pairs)
    # Каждая пара должна заканчиваться на разрешённые квоты (USDT и т.п.)
    assert all(any(p.endswith(q) for q in QUOTE_ASSETS) for p in pairs)

    logging.info("Found: %d pairs. Examples: %s", len(pairs), pairs[:10])
