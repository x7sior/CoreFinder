import numpy as np


def generate_game(n: int,
                  superadditive: bool = False,
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Генерирует случайную характеристическую функцию.

    Параметры
    ---------
    n             : число игроков
    superadditive : применять ли суперадиитивность (False по умолчанию)
    rng           : np.random.Generator

    Возвращает
    ----------
    v : np.ndarray[2^n]
    """
    if rng is None:
        rng = np.random.default_rng()

    size     = 1 << n
    masks    = np.arange(size, dtype=np.int32)
    bits     = np.arange(n, dtype=np.int32)
    popcount = ((masks[:, None] >> bits[None, :]) & 1).sum(axis=1)

    v      = np.zeros(size, dtype=np.float64)
    v[1:]  = rng.random(size - 1) * popcount[1:].astype(np.float64)

    if superadditive:
        _make_superadditive_inplace(v, popcount)

    return v


def _make_superadditive_inplace(v: np.ndarray, popcount: np.ndarray) -> None:
    """Суперадиитивность через битовый перебор подмножеств"""
    order = np.argsort(popcount[1:]) + 1
    for mask in order:
        if popcount[mask] < 2:
            continue
        sub = (mask - 1) & mask
        while sub > 0:
            complement = mask ^ sub
            if sub < complement:  # каждую пару считаем один раз
                val = v[sub] + v[complement]
                if val > v[mask]:
                    v[mask] = val
            sub = (sub - 1) & mask
