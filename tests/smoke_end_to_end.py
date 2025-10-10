import logging
import pytest
from pybit.unified_trading import HTTP

from data_loader import set_client as set_dl_client
from pair_finder import set_client as set_pf_client, get_candidate_pairs
from pair_selector import evaluate_pairs
from config import BYBIT_API_KEY, BYBIT_API_SECRET, INTERVAL

logging.basicConfig(level=logging.INFO)

@pytest.mark.smoke
def test_end_to_end_top5():
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_dl_client(client)
    set_pf_client(client)

    pairs = get_candidate_pairs(force_refresh=True)[:15]
    if not pairs:
        pytest.skip("Finder вернул пусто (лимиты API?)")

    top = evaluate_pairs(
        symbols=pairs,
        interval=INTERVAL,
        top_n=5,
        return_scores=True,
        debug=True,
        window=200,
        save_results=True,
    )
    assert isinstance(top, list) and len(top) > 0
    sym, score = top[0]
    assert isinstance(sym, str)
    assert isinstance(score, float)
