# CoreFinder
Разработка и тестирование алгоритмов построения решения из С-ядра для кооперативной игры (ВКР)

## Установка
```bash
pip install -r requirements.txt
```

## Структура репозитория

**theory.pdf** — текст из ВКР для ознакомления с теоретической частью.

**CoreFinder.py** — программная реализация алгоритмов: `CoreFinder`, `CoreFinderOpt`, `CoreFinderAlt`.

**other_algorithms.py** — вспомогательные алгоритмы: `min_sum_core_point`, `shapley_value`, `nucleolus`.

**game_generator.py** — генератор случайных и супераддитивных игр: `generate_game`.

**test_prepare.py** — функции для проведения тестов: `pregenerate_games`, `precompute`, `compute`, `compute_distance`.

**tests.ipynb** — ноутбук с тестами и графиками.