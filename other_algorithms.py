import numpy as np
from math import factorial
from scipy.optimize import linprog


def min_sum_core_point(n: int, v: np.ndarray) -> np.ndarray:
    """
    Нахождение точки C-ядра с минимальной суммой компонент.
    ---------
    Используется LP-минимизация с ограничениями в виде полной характеристической функции, кроме v(N)
    
    Параметры
    ---------
    n : int              — число игроков
    v : np.ndarray[2^n]  — характеристическая функция (битмаски)

    Возвращает
    ----------
    x : np.ndarray[n]  — точка ядра с минимальной суммой,
                         или None если ядро пусто
    """
    # Все непустые коалиции кроме N
    N            = (1 << n) -1
    proper_masks = np.array([m for m in range(1, N)], dtype=np.int32)

    bits = np.arange(n, dtype=np.int32)

    # Матрица ограничений A[j, i] = 1 если игрок i в proper_masks[j]
    A = ((proper_masks[:, None] >> bits[None, :]) & 1).astype(np.float64)

    # min  1^T x
    # s.t. A @ x >= v[proper_masks]   =>   -A @ x <= -v[proper_masks]
    result = linprog(
        c      = np.ones(n),
        A_ub   = -A,
        b_ub   = -v[proper_masks],
        bounds = [(None, None)] * n,
        method = "highs",
        options = {"disp": False},
    )

    if result.status == 0:
        return result.x
    return None


def shapley_value(n: int, v: np.ndarray) -> np.ndarray:
    """
    Вычисление вектора Шэпли для кооперативной игры.
    ---------
    Параметры
    ---------
    n : int              — число игроков
    v : np.ndarray[2^n]  — характеристическая функция (битовые маски)

    Возвращает
    ----------
    phi : np.ndarray[n]  — вектор Шэпли, phi[i] — выигрыш игрока i+1

    Формула:
        phi_i = sum_{{S: i in S}} w(|S|) * (v(S) - v(S \\ {{i}}))
        w(s)  = (s-1)! * (n-s)! / n!
    """
    fn      = factorial(n)
    size    = 1 << n
    masks   = np.arange(size, dtype=np.int32)
    bits    = np.arange(n,    dtype=np.int32)

    # indicator[mask, i] = 1 если игрок i в коалиции mask
    indicator = ((masks[:, None] >> bits[None, :]) & 1).astype(np.float64)

    # popcount[mask] = размер коалиции
    popcount = indicator.sum(axis=1).astype(np.int32)

    # weights[mask] = (|S|-1)! * (n-|S|)! / n!
    weights = np.zeros(size, dtype=np.float64)
    for mask in range(1, size):
        s = int(popcount[mask])
        weights[mask] = factorial(s - 1) * factorial(n - s) / fn

    # without_i[i, mask] = mask без игрока i (если i в mask, иначе 0)
    without_i = np.zeros((n, size), dtype=np.int32)
    for i in range(n):
        bit    = 1 << i
        with_i = np.where((masks & bit) != 0)[0]
        without_i[i, with_i] = with_i - bit

    # phi[i] = sum_{S: i in S} weights[S] * (v[S] - v[S \ {i}])
    nonempty = np.arange(1, size, dtype=np.int32)
    phi      = np.zeros(n, dtype=np.float64)
    for i in range(n):
        with_i    = nonempty[indicator[nonempty, i] == 1]
        phi[i]    = np.dot(weights[with_i], v[with_i] - v[without_i[i, with_i]])

    return phi


def nucleolus(v: np.ndarray, n: int, tol: float = 1e-9) -> np.ndarray:
    """
    Вычисление N-ядра кооперативной игры.

    Параметры
    ---------
    v   : np.ndarray[2^n]  — характеристическая функция (битовые маски)
    n   : int              — число игроков
    tol : float            — допуск для определения насыщенных коалиций

    Возвращает
    ----------
    x : np.ndarray[n]  — N-ядро (игроки индексируются с 0)

    N-ядро лексикографически минимизирует вектор эксцессов:
        e(S, x) = v(S) - sum_{i in S} x_i

    отсортированных по убыванию, при ограничении sum x_i = v(N).

    Алгоритм: последовательность LP задач.
    На каждом шаге:
    1. Минимизируем максимальный эксцесс epsilon
    2. Фиксируем коалиции где эксцесс = epsilon (они "насыщены")
    3. Повторяем на оставшихся коалициях

    """
    N    = (1 << n) - 1
    size = 1 << n

    # Индикаторная матрица
    masks    = np.arange(size, dtype=np.int32)
    bits     = np.arange(n,    dtype=np.int32)
    indicator = ((masks[:, None] >> bits[None, :]) & 1).astype(np.float64)

    # Активные коалиции: все непустые кроме N и пустой
    active = list(range(1, N))

    # Переменные LP: [x_0, ..., x_{n-1}, epsilon]
    # Размерность: n + 1

    x = np.zeros(n, dtype=np.float64)

    while active:
        m = len(active)
        active_arr = np.array(active, dtype=np.int32)

        # Матрица ограничений для эксцессов:
        # e(S, x) = v(S) - x(S) <= epsilon
        # => -x(S) - epsilon <= -v(S)
        # => -indicator[S] @ x - epsilon <= -v[S]
        # Переменные: [x_0,...,x_{n-1}, epsilon]
        # A_ub shape: (m, n+1)
        A_ub = np.hstack([
            -indicator[active_arr], # -x(S)
            -np.ones((m, 1))    # -epsilon
        ])
        b_ub = -v[active_arr]

        # Ограничение суммы: sum x_i = v(N)
        A_eq = np.hstack([
            np.ones((1, n)),
            np.zeros((1, 1))
        ])
        b_eq = np.array([v[N]])

        # min epsilon
        c = np.zeros(n + 1)
        c[-1] = 1.0

        result = linprog(
            c      = c,
            A_ub   = A_ub,
            b_ub   = b_ub,
            A_eq   = A_eq,
            b_eq   = b_eq,
            bounds = [(None, None)] * n + [(None, None)],
            method = "highs",
            options = {"disp": False},
        )

        if result.status != 0:
            break

        x       = result.x[:n]
        epsilon = result.x[-1]

        # Находим насыщенные коалиции: e(S, x) ≈ epsilon
        excesses = v[active_arr] - indicator[active_arr] @ x
        saturated = [active_arr[i] for i in range(m)
                     if abs(excesses[i] - epsilon) <= tol]

        if not saturated:
            break

        # Убираем насыщенные из активных
        saturated_set = set(saturated)
        active = [s for s in active if s not in saturated_set]

    return x