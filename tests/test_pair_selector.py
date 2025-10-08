import logging
import pytest
from pybit.unified_trading import HTTP

from data_loader import set_client as set_dl_client
from pair_finder import get_candidate_pairs, set_client as set_pf_client
from pair_selector import evaluate_pairs
from config import BYBIT_API_KEY, BYBIT_API_SECRET, INTERVAL

logging.basicConfig(level=logging.INFO)


@pytest.mark.smoke
def test_pair_selector_smoke():
    """
    Смоук-тест: проверяем, что пайплайн finder → selector работает без ошибок.
    """
    # === 1️⃣ Инициализация клиента ===
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_dl_client(client)
    set_pf_client(client)

    # === 2️⃣ Получаем кандидатов ===
    pairs = get_candidate_pairs(force_refresh=True)[:10]
    if not pairs:
        pytest.skip("⚠️ No pairs fetched (API limit or empty response).")

    logging.info(f"Candidate pairs ({len(pairs)}): {pairs}")

    # === 3️⃣ Запускаем селектор ===
    top = evaluate_pairs(
        symbols=pairs,
        interval=INTERVAL,
        top_n=5,
        return_scores=True,
        debug=True,
        window=200,
        save_results=True,   # сохранит outputs/top_pairs_<ts>.csv
    )

    # === 4️⃣ Проверяем базовые инварианты ===
    assert isinstance(top, list), "evaluate_pairs должен вернуть list"
    assert len(top) > 0, "Результат не должен быть пустым"

    sym, score = top[0]
    assert isinstance(sym, str), "Имя тикера должно быть строкой"
    assert isinstance(score, float), "Скор должен быть float"
    assert -10.0 < score < 50.0, f"Неверный диапазон score: {score}"

    logging.info(f"✅ Selector smoke OK. Top pairs: {top}")
