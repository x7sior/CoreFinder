from collections.abc import Sequence
class CoreFinder:
    """
    Базовый алгоритм поиска предрешения игры.

    Parameters
    ----------
    n : sequence or int
        Множество игроков. Задаётся последовательностью идентификаторов либо целым числом — тогда игроки нумеруются от 1 до n.
    v : dict
        Характеристическая функция {последовательность из идентификаторов коалиции: выигрыш}.
    precision : int, optional
        Число знаков после запятой при выводе результата. По умолчанию 10.
    eps : int, optional
        Погрешность при сравнении. Применяется как 1e-eps. По умолчанию 10.

    Attributes
    ----------
    x : dict
        Предрешение игры {номер игрока от 0 до n-1: выигрыш}
    x_N : float
        Сумма компонент предрешения игры.
    delta_v : float
        Сумма дополнительных выплат.
        
    Methods
    -------
    get_x()
        Возвращает вычисленное предрешение игры в виде словаря {идентификатор игрока: выигрыш}.
    get_imputation(extra_imputation_form=None)
        Вычисляет решение игры.
    check_excess(check="imputation", get_excess: bool=False, print_excesses: bool=False)
        Проверяет устойчивость относительно коалиций.

    Examples
    --------
    >>> cf = CoreFinder(['0', '1', '2'], {'0': 0, '1': 0, '2': 0, '01': 10, '02': 20, '12': 30, '012': 50})
    >>> cf = CoreFinder('123', {'1': 0, '2': 0, '3': 0, '12': 10, '13': 20, '23': 30, '123': 50})
    >>> cf = CoreFinder(3, {frozenset([1]): 0, frozenset([2]): 0, frozenset([3]): 0, frozenset([1, 2]): 10, frozenset([1, 3]): 20, frozenset([2, 3]): 30, frozenset([1, 2, 3]): 50})
    >>> cf.get_x()
    {'1': 0.0, '2': 10.0, '3': 20.0}
    >>> cf.x_N
    30.0
    >>> cf.get_imputation()
    {'1': 6.6666666667, '2': 16.6666666667, '3': 26.6666666667}
    >>> cf.get_imputation([0.2, 0.3, 0.5])
    {'1': 4.0, '2': 16.0, '3': 30.0}
    >>> cf.check_excess()
    {'in_core': True,
    'excess': {1: -4.0, 2: -16.0, 4: -30.0, 3: -10.0, 5: -14.0, 6: -16.0}}
    >>> cf.check_excess(get_excess=True)
        {'in_core': True,
    'excess': {frozenset({'1'}): -4.0,
    frozenset({'2'}): -16.0,
    frozenset({'3'}): -30.0,
    frozenset({'1', '2'}): -10.0,
    frozenset({'1', '3'}): -14.0,
    frozenset({'2', '3'}): -16.0}}

    Raises
    ------
    AssertionError
        Если `v` не является словарём.
    ValueError
        Если `v` задана не для всех коалиций.
    KeyError
        Если коалиция в `v` содержит игрока, отсутствующего в `n`.
    """
    
    def __init__(
            self,
            n: Sequence | int,
            v: dict,
            precision: int = 10,
            eps: int = 10
            ):
        
        self._add_n(n)
        self._add_v(v)
        self.precision = precision
        self.eps = 10 ** (-eps)
        self._run()
        self.x = {i: round(self.x[i], self.precision) for i in range(self.n)}
        self.x_N = sum(self.x.values())
        self.delta_v = self.v[self.n][(1 << self.n) - 1] - self.x_N
        self.imputation = None

    def _add_n(self, n):
        
        if isinstance(n, int):
             n = [i for i in range(1, n + 1)]

        self.n = len(n)
        self.players = {n[i]: i for i in range(self.n)}

    def _add_v(self, v: dict):
        
        assert isinstance(v, dict), "Функция выигрыша должна быть словарем"

        if len(v) != 2**self.n - 1:
            raise ValueError("Функция выигрыша задана не для всех коалиций")
        
        try:
            self.v = {i: dict() for i in range(1, self.n + 1)}
            for coalition in v:
                mask = 0
                for player in coalition:
                    mask |= 1 << (self.players[player])
                self.v[len(coalition)][mask] = v[coalition]
        except KeyError as e:
            raise KeyError(f"Игрок {e} отсутствует в множестве игроков")

    def get_x(self):
        """
        Предрешение игры.

        Returns
        -------
        dict
            {идентификатор игрока: выигрыш}.
        """
        return {i: self.x[self.players[i]] for i in self.players}

    def get_imputation(self, extra_imputation_form: Sequence = None):
        """
        Вычисляет решение.
        
        Parameters
        ----------
        extra_imputation_form : sequence, optional
            Определяет пропорции распределения дополнительных выплат в порядке нумерации игроков. По умолчанию равномерное распределение.

        Returns
        -------
        dict
            Вычисленное решение игры {идентификатор игрока: выигрыш}.

        Notes
        -----
        Если задан `extra_imputation_form` или `imputation` is None,
        пересчитывает и задаёт атрибут `imputation` как {номер игрока: выигрыш},
        где номер — позиция игрока от 0 до n-1.
        """
        if extra_imputation_form or (self.imputation is None):
            if extra_imputation_form is None:
                extra_imputation_form = [1/self.n] * self.n

            self.imputation = {
                i: round(self.x[i] + self.delta_v * extra_imputation_form[i], self.precision)
                for i in range(self.n)
                }

        return {i: self.imputation[self.players[i]] for i in self.players}

    def check_excess(self,
                     check = 'imputation',
                     get_excess: bool = False,
                     print_excesses: bool = False) -> dict:
        """
        Проверяет устойчивость относительно всех коалиций, кроме максимальной.

        Parameters
        ----------
        check : {'imputation', 'x'} or sequence, optional
            Задаёт проверяемое распределение. По умолчанию 'imputation'.
        get_excess : bool, optional
            Если True, коалиции в 'excess' возвращаются как frozenset идентификаторов, иначе как битмаски. По умолчанию False.
        print_excesses : bool, optional
            Вывести ли эксцессы на экран. По умолчанию False.

        Returns
        -------
        dict
            Словарь с ключами:
            - 'in_core' : bool — сохраняется ли устойчивость на всех коалициях.
            - 'excess' : dict — {коалиция: эксцесс}.

        Raises
        ------
        ValueError
            Если `check` принимает недопустимое значение или его длина не равна количеству игроков.
        """
        match check:
            case "imputation":
                if self.imputation is None:
                    self.get_imputation()

                imputation = self.imputation
            case "x":
                 imputation = self.x
            case Sequence():
                 if len(check) != self.n:
                     raise ValueError("Длина последовательности пропорций должна быть равна количеству игроков")
                 else:
                     imputation = {i: check[i] for i in range(self.n)}
            case _:
                raise ValueError("Недопустимое значение параметра 'check'")

        # вычислим суммы игроков
        sum_x = [0] * (1 << self.n)

        for coalition in range(1, 1 << self.n):
            lsb = coalition & -coalition
            sum_x[coalition] = sum_x[coalition ^ lsb] + imputation[lsb.bit_length() - 1]

        # вычислим эксцессы
        excess = {}

        in_core = True
        for k in range(1, self.n):
            vk = self.v[k]

            for coalition in vk:
                values = vk[coalition] - sum_x[coalition]
                excess[coalition] = values

                if values > self.eps * 10:
                    in_core = False

        if get_excess or print_excesses:
            get_players = self._get_players
            players_dict = {i: player for player, i in self.players.items()}
            excess_set = {
                frozenset(players_dict[i] for i in get_players(coalition)): excess[coalition]
                for coalition in excess}

            if print_excesses:
                for coalition in excess_set:
                    print(f"{list(coalition)}: {excess_set[coalition]}")

            if get_excess:
                return {"in_core": in_core, "excess": excess_set}
            
        return {"in_core": in_core, "excess": excess}

    @staticmethod
    def _get_players(mask):
        players = []
        while mask:
            lsb = mask & -mask
            players += [lsb.bit_length() - 1]
            mask ^= lsb
        return players

    def _run(self):
        get_players = self._get_players
        x = {i: self.v[1][1 << i] for i in range(self.n)}

        for k in range(self.n, 2, -1):
            k_minus = k-1
            vk = self.v[k]
            vkm = self.v[k_minus]

            for main_coalition in vk:
                players = get_players(main_coalition)
                coalitions = [vkm[main_coalition ^ (1 << i)] for i in players]
                term = sum(coalitions) / k_minus  # общий член в формуле

                for i in range(k):
                    x[players[i]] = max(term - coalitions[i], x[players[i]])
                
        self.x = x


class CoreFinderOpt(CoreFinder):
    """
    Оптимизированный алгоритм поиска предрешения.

    See Also
    --------
    CoreFinder : Базовый класс, содержит описание параметров и методов.
    """
    def _grand_coalition_x(self):
        # локальные переменные
        n = self.n
        x = {}
        main_coalition = (1 << n) - 1
        k = n - 1
        vk = self.v[k]

        # сохраняем неотрицательные выигрыши одиночных коалиций
        v1 = {
            i: max(self.v[1][i], 0)
              for i in self.v[1]} 

        # урезаем выигрыши коалиций на значения одиночных
        sum_single_coalitions = sum([v1[1 << i] for i in range(n)]) # сумма выигрышей одиночных коалиций
        coalitions = [
            max(vk[main_coalition ^ (1 << i)] - sum_single_coalitions + v1[1 << i], 0) 
            for i in range(n)] # список значений урезанных предграндовых коалиций
        
        # вычисляем предрешение
        term = sum(coalitions) / k # общий член в формуле
        x = [
            max(term - coalitions[i], 0) + v1[1 << i] # к полученному значению возвращаем собственный выигрыш игрока
            for i in range(n)]
        
        # считаем суммы игроков
        ful_sum_x = sum(x[i] for i in range(n)) # сумма всех игроков
        sum_x = {
            (1 << i) ^ main_coalition: ful_sum_x - x[i]
            for i in range(n)} # сохраняем суммы игроков для коалиций размера n-1
        
        # сохраняем результаты
        self.x = x
        self.sum_x = sum_x

    def _run(self):
        self._grand_coalition_x()
        get_players = self._get_players
        
        # функция для поиска неустойчивых вложенных коалиций
        def exc_subcoalitions(k_plus, exc_coalitions, fixed_coalition, players_keys):
            coalitions = [] # положительные эксцессы
            coalitions_keys = [] # маски неустойчивых коалиций
            zero_excess_keys = [] # индексы игроков, без которых зафиксированная коалиция становится устойчивой

            for i in range(k_plus):
                mask = fixed_coalition ^ (1 << players_keys[i]) # маска коалиции без i-го игрока
                if mask in exc_coalitions:
                    coalitions += [exc_coalitions[mask]]
                    coalitions_keys += [mask]
                else:
                    zero_excess_keys += [i]
            return coalitions_keys, zero_excess_keys, coalitions

        # функция для построения фиксируемой коалиции
        def find_main_coalition(k_plus, exc_coalitions, exc_players_keys):
            most_exc_coalition_mask = max(exc_coalitions, key=exc_coalitions.get) # маска самой недовольной коалиции

            for player in exc_players_keys:
                if most_exc_coalition_mask & (1 << player) == 0: # если игрок не входит в самую недовольную коалицию
                    changed_coalition = most_exc_coalition_mask | (1 << player) # добавляем игрока в коалицию
                    changed_keys = [player] + get_players(most_exc_coalition_mask) # индексы всех игроков из построенной коалиции
                    return (
                        exc_subcoalitions(k_plus, exc_coalitions, changed_coalition, changed_keys)
                        + (changed_keys,)
                        )

        # локальные переменные
        eps = self.eps
        n = self.n
        x = self.x
        old_sum_x = self.sum_x  # старые суммы игроков (коалиций размера k+1)
        for k in range(n - 2, 1, -1):
            k_plus = k+1
            vk = self.v[k]
            total_extra_x = {i: 0 for i in range(n)} # добавочная сумма выплат к предрешению каждого игрока

        # считаем суммы и ищем неустойчивые коалиции
            new_sum_x = dict() # новые суммы игроков (коалиций размера k)
            exc_coalitions = dict() # эксцессы неустойчивых коалиций (размера k)
            exc_players = {i: 0 for i in range(n)} # сумма эксцессов по всем неустойчивым коалициям, в которые входит игрок

            for coalition in vk:
                # ищем коалицию включащую выбранную, чтобы вычислить её сумму и эксцесс
                for i in range(n):
                    bigger_coalition = coalition | (1 << i)
                    if bigger_coalition in old_sum_x:
                        new_sum_x[coalition] = old_sum_x[bigger_coalition] - x[i]
                        break
                excess = vk[coalition] - new_sum_x[coalition]

                # сохраняем эксцессы устойчивых коалиций и входящих в них игроков
                if excess > eps:
                    exc_coalitions[coalition] = excess
                    for player in get_players(coalition):
                        exc_players[player] += excess

        # дополняем предрешение и пересчитываем суммы на основе неустойчивых коалиций
            while exc_coalitions:
                exc_players_keys = sorted(exc_players, key=exc_players.get, reverse=True) # индекcы самых неустойчивых игроков
                extra_x = {} # добавочные выплаты к предрешению для каждого игрока в текущей итерации

                # случай, когда осталасть одна неустойчивая коалиция
                if len(exc_coalitions) == 1:
                    exc_coalitions = None
                    term = exc_players[exc_players_keys[0]] / k # общий член в формуле предрешения
                    
                    for player in exc_players_keys[:k]:
                        x[player] += term
                        extra_x[player] = term

                # более одной неустойчивой коалиции
                else:
        # строим маску коалиции по наиболее неустойчивым игрокам и сохраняем эксцессы коалиций, вложенных в неё
                    main_coalition = 1 << exc_players_keys[0] # добавляем первого игрока 
                    for i in range(1, k_plus):
                        main_coalition |= 1 << exc_players_keys[i] # добавляем всех остальных

                    coalitions_keys, zero_excess_keys,coalitions = exc_subcoalitions(
                        k_plus, exc_coalitions, main_coalition, exc_players_keys) # эксцессы вложенных коалиций

        # если среди вложенных коалиций нет ни одной неустойчивой, то строим маску коалиции, у которой они точно есть
                    if not coalitions:
                        coalitions_keys, zero_excess_keys, coalitions, exc_players_keys = find_main_coalition(
                            k_plus, exc_coalitions, exc_players_keys)

        # считаем дополнение к предрешению
                    term = sum(coalitions) / k  # общий член в формуле
                    idx = 0 # индекс текущей коалиции, среди неустойчивых

                    for i in range(k_plus):
                        if i in zero_excess_keys:
                            x_value = term
                        else:
                            x_value = term - coalitions[idx]
                            idx += 1

                        if x_value > eps:
                            player = exc_players_keys[i]
                            x[player] += x_value
                            extra_x[player] = x_value

        # удаляем использованные эксцессы
                    for key in coalitions_keys:
                        del exc_coalitions[key]

        # пересчитываем эксцессы оставшихся коалиций и удаляем устойчивые
                    for player in extra_x:
                        del_excesses = []

                        for coalition in exc_coalitions:
                            if coalition & (1 << player):
                                excesses_value = exc_coalitions[coalition] - extra_x[player]

                                if excesses_value > eps:
                                    exc_coalitions[coalition] = excesses_value
                                else:
                                    del_excesses += [coalition]

                        for coalition in del_excesses:
                            del exc_coalitions[coalition]

        # пересчитываем суммы эксцессов для игроков
                    exc_players = {i: 0 for i in range(n)}

                    for coalition in exc_coalitions:
                        excess = exc_coalitions[coalition]

                        for player in get_players(coalition):
                            exc_players[player] += excess

        # пересчитываем суммы добавочных выплат каждого игрока
                for player in extra_x:
                    total_extra_x[player] += extra_x[player]
            
        # пересчитываем суммы игроков
            for player in total_extra_x:
                if total_extra_x[player] > eps: # если сумма добавочных выплат игрока больше нуля
                    for coalition in new_sum_x: # среди всех сумм
                        if coalition & (1 << player): # находим коалиции, в которые входит игрок
                            new_sum_x[coalition] += total_extra_x[player] # и добавляем сумму добавочных выплат
            old_sum_x = new_sum_x

        # сохраняем результат
        self.x = x



class CoreFinderAlt(CoreFinder):    
    """
    Альтернативный алгоритм поиска предрешения.

    Parameters
    ----------
    early_stop : bool, optional
        Если True, останавливает вычисление при попадании решения в C-ядро. По умолчанию True.

    See Also
    --------
    CoreFinder : Базовый класс, содержит описание параметров и методов.
    """
    def __init__(
            self,
            n: Sequence | int,
            v: dict,
            early_stop: bool = True,
            precision: int = 10,
            eps: int = 10
            ):
        self.early_stop = early_stop
        super().__init__(n, v, precision, eps)

    def _run(self):
        # локальные переменные
        iteration = self._iteration
        get_players = self._get_players
        early_stop = self.early_stop
        eps = self.eps
        n = self.n
        v = self.v
        v1 = v[1]
        vn = list(v[n].values())[0]

        # стартовые значения
        x = {i: v1[1 << i] if v1[1 << i] > 0 else 0 for i in range(n)}
        x_min = x
        best_ful_sum_x = float("inf")
        exc_coalitions = {i: v[i].copy() for i in range(2, n)}

        # проходим по эксцессам коалиций, обновляя список эксцессов
        while exc_coalitions:
            del_k = []  # список пустых групп одного размера

            for k in exc_coalitions:
                amount = 0  # счётчик коалиций в одной группе
                vk = exc_coalitions[k]
                del_exc_coalitions = [] # список лишних коалиций

                for coalition in vk:
                    amount += 1
                    for player in get_players(coalition):
                        vk[coalition] -= x_min[player]

        # удаляем лишние группы и коалиции из списка эксцессов
                    if vk[coalition] <= eps:
                        del_exc_coalitions += [coalition]

                for coalition in del_exc_coalitions:
                    del vk[coalition]
                    amount -= 1

                if amount == 0:
                    del_k += [k]

            for k in del_k:
                del exc_coalitions[k]

        # считаем минимальное и максималное предрешение по эксцессам
            x_min, x_max = iteration(exc_coalitions)

        # сохраняем лучшее предрешение
            tmp_x = {i: x[i] + x_max[i] for i in range(n)}
            ful_sum_x = sum(tmp_x.values())

            if ful_sum_x <= best_ful_sum_x:
                self.x = tmp_x
                best_ful_sum_x = ful_sum_x

                if early_stop and best_ful_sum_x <= vn:
                    break

            x = {i: x[i] + x_min[i] for i in range(n)}

    def _iteration(self, exc_coalitions):
        # локальные переменные
        get_players = self._get_players
        eps = self.eps
        x_min = {i: 0 for i in range(self.n)}
        x_max = {i: 0 for i in range(self.n)}

        for k_minus in exc_coalitions:
            k = k_minus + 1
            vk = self.v[k]
            vkm = exc_coalitions[k_minus]

        # фиксируем коалицию, по которой будет построено предрешение
            for main_coalition in vk:
                players = get_players(main_coalition)
                coalitions = []
                zero_excess_keys = []

        # сохраняем эксцессы вложенных коалиций
                for i in range(k):
                    coalition_mask = main_coalition ^ (1 << players[i])

                    if coalition_mask in vkm:
                        coalitions += [vkm[coalition_mask]]
                    else:
                        zero_excess_keys += [i]

        # считаем минимальные и максимальные предрешения по эксцессам
                if len(coalitions):
                    term = sum(coalitions) / k_minus  # общий член в формуле
                    idx = 0    # индекс текущей коалиции среди неустойчивых

                    for i in range(k):
                        if i in zero_excess_keys:
                            x_value = term
                        else:
                            x_value = term - coalitions[idx]
                            idx += 1

                        if x_value > eps:
                            player = players[i]
                            x_max[player] = max(x_value, x_max[player])

                            if x_min[player]:
                                x_min[player] = min(x_value, x_min[player])
                            else:
                                x_min[player] = x_value
                                
        return x_min, x_max
    

