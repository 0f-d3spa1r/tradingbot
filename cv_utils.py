# cv_utils.py
import numpy as np
from typing import Iterator, Tuple

def purged_cv_splits(n: int, n_splits: int = 5, embargo: int = 10) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Генерирует (train_idx, test_idx) для временных рядов.
    embargo: сколько наблюдений исключить после тестового окна из train (purge).
    """
    fold_size = n // (n_splits + 1)
    for i in range(n_splits):
        test_start = (i + 1) * fold_size
        test_end   = test_start + fold_size
        test_idx   = np.arange(test_start, min(test_end, n))

        train_end  = max(0, test_start - embargo)
        train_idx  = np.arange(0, train_end)

        yield train_idx, test_idx
