from collections.abc import Sequence
class CoreFinder:    
    def __init__(
            self,
            n: Sequence | int = None,
            v: dict = None,
            precision: int = 10,
            eps: int = 10
            ):
        
        self._add_n(n)
        self._add_v(v)
        self.precision = precision    # порядок округления результата (число знаков после запятой)
        self.eps = 10 ** (-eps)       # погрешность при сравнения
        self._run()
        self.x = {i: round(self.x[i], self.precision) for i in range(self.n)}
        self.x_N = sum(self.x.values())
        self.delta_v = self.v[self.n][(1 << self.n) - 1] - self.x_N
        self.imputation = None

    def _add_n(self, n):
        if n is None:
            raise ValueError("Множество игроков не задано")
        
        if isinstance(n, int):
             n = [i for i in range(1, n + 1)]

        self.n = len(n)
        self.players = {n[i]: i for i in range(self.n)}

    def _add_v(self, v: dict):
        if v is None:
            raise ValueError("Функция выигрыша не задана")
        
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
        return {i: self.x[self.players[i]] for i in self.players}

    @staticmethod
    def _get_players(mask):
        players = []
        while mask:
            lsb = mask & -mask
            players += [lsb.bit_length() - 1]
            mask ^= lsb
        return players

    def get_imputation(self, extra_imputation_form: Sequence = None):
        if extra_imputation_form or (self.imputation is None):
            if extra_imputation_form is None:
                extra_imputation_form = [1/self.n] * self.n

            self.imputation = {
                i: round(self.x[i] + self.delta_v * extra_imputation_form[i], self.precision)
                for i in range(self.n)
                }

        return {i: self.imputation[self.players[i]] for i in self.players}

    def check_excess(self,
                     check = "imputation",
                     get_excess: bool = False,
                     print_excesses: bool = False) -> dict:
        """
        Проверяет условия v(S) <= sum(x[i] for i in S) для всех коалиций S, и возвращает словарь с результатами проверки и эксцессами.

        Параметры
        ---------
        check          : "imputation" | "x"  | Sequence -- задать проверяемое распределение (по умолчанию "imputation")
        get_excess     : возвращать ли эксцессы с коалициями в виде множеств(False по умолчанию)
        print_excesses : вывести ли эксцессы на экран (False по умолчанию)

        Возвращает dict: "in_core", "excess" (по умолчанию словарь эксцессов с коалициями в виде битовых масок, если get_excess=False, и в виде множеств игроков, если get_excess=True)
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

        # посчитаем суммы игроков
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
    def __init__(
            self,
            n: Sequence | int = None,
            v: dict = None,
            precision: int = 10,
            eps: int = 10
            ):
        
        super().__init__(n, v, precision, eps)

    def _grand_coalition_x(self):
        # локальные переменные
        n = self.n
        x = {}
        main_coalition = (1 << n) - 1
        k = n - 1
        vk = self.v[k]

        # сохраняем неотрицательные выигрыши одиночных коалиций
        v1 = {i: max(self.v[1][i], 0) for i in self.v[1]} # сохраняем выигрыши одиночных коалиций

        # урезаем выигрыши коалиций на значения одиночных
        sum_single_coalitions = sum([v1[1 << i] for i in range(n)]) # сумма выигрышей одиночных коалиций
        coalitions = [
            max(vk[main_coalition ^ (1 << i)] - sum_single_coalitions + v1[1 << i], 0) 
            for i in range(n)
            ] # список значений урезанных предграндовых коалиций
        
        # вычсиляем предрешение
        term = sum(coalitions) / k # общий член в формуле

        for i in range(n):
            x[i] = max(term - coalitions[i], 0) + v1[1 << i] # добавляем собственный выигрыш, что ранее был отнят у коалиций
        
        # считаем суммы игроков
        ful_sum_x = sum(x[i] for i in range(n)) # сумма всех игроков
        sum_x = {(1 << i) ^ main_coalition: ful_sum_x - x[i] for i in range(n)} # сохраняем суммы игроков для коалиций размера n-1
        
        # сохраняем результаты
        self.x = x
        self.sum_x = sum_x

    def _run(self):
        self._grand_coalition_x()
        get_players = self._get_players
        
        # функция для поиска "недовольных" коалиций, которые являются подкоалициями фиксированной коалиции
        def exc_subcoalitions(k_plus, exc_coalitions, fixed_coalition, players_keys):
            coalitions_keys = []    # сохраняем маски вложенных коалиций
            zero_excess_keys = []   # сохраняем индексы коалиций в списке, у которых эксцессы нулевые
            coalitions = []

            for i in range(k_plus):
                mask = fixed_coalition ^ (1 << players_keys[i]) # строим маску коалиции без i-го игрока
                if mask in exc_coalitions:
                    coalitions += [exc_coalitions[mask]] # сохраняем положительные эксцессы коалиций без i-го игрока
                    coalitions_keys += [mask]
                else:
                    zero_excess_keys += [i]
            return coalitions_keys, zero_excess_keys, coalitions

        # функция для поиска основной коалиции, которая может быть построена на основе "недовольных" коалиций
        def find_main_coalition(k_plus, exc_coalitions, exc_players_keys):
            most_exc_coalition_mask = max(exc_coalitions, key=exc_coalitions.get)  # маска самой недовольной коалиции
            for player in exc_players_keys:
                if most_exc_coalition_mask & (1 << player) == 0: # если игрок не входит в эту коалицию
                    changed_coalition = most_exc_coalition_mask | (1 << player) # добавляем игрока в эту коалицию
                    changed_keys = [player] + get_players(most_exc_coalition_mask)
                    return (
                        exc_subcoalitions(k_plus, exc_coalitions, changed_coalition, changed_keys)
                        + (changed_keys,)
                        )

        # локальные переменные
        eps = self.eps
        n = self.n
        x = self.x
        old_sum_x = self.sum_x  # словарь для хранения старых сумм игроков (коалиций размера k+1)
        for k in range(n - 2, 1, -1):
            k_plus = k+1
            vk = self.v[k]
            total_extra_x = {i: 0 for i in range(n)} # словарь для хранения суммы дополнений к предрешению для каждого игрока

        # считаем суммы и ищем "недовольные" коалиции
            new_sum_x = dict() # словарь для хранения новых сумм игроков (коалиций размера k)
            exc_coalitions = dict() # словарь для хранения эксцессов "недовольных" коалиций
            exc_players = {i: 0 for i in range(n)} # словарь для подсчёта суммы эксцессов всех "недовольных" коалиций, в которые входит игрок

            for coalition in vk:
                # ищем коалицию включащую выбранную, чтобы быстрее посчитать её сумму и эксцесс
                for i in range(n):
                    bigger_coalition = coalition | (1 << i)
                    if bigger_coalition in old_sum_x:
                        new_sum_x[coalition] = old_sum_x[bigger_coalition] - x[i]
                        break
                excess = vk[coalition] - new_sum_x[coalition]

                # сохраняем положительные эксцессы для коалиций и входящих в них игроков
                if excess > eps:
                    exc_coalitions[coalition] = excess
                    for player in get_players(coalition):
                        exc_players[player] += excess

        # дополняем предрешение и пересчитываем суммы на основен "недовольных коалиций"
            while exc_coalitions:
                exc_players_keys = sorted(exc_players, key=exc_players.get, reverse=True) # индекcы самых "недовольных" игроков
                extra_x = {} # словарь для хранения дополнений к предрешению для каждого игрока в текущей итерации

                # случай, когда осталасть одна "недовольная" коалиция
                if len(exc_coalitions) == 1:
                    exc_coalitions = None
                    term = exc_players[exc_players_keys[0]] / k # общий член в формуле предрешения
                    
                    for player in exc_players_keys[:k]:
                        x[player] += term
                        extra_x[player] = term

                # более одной "недовольной" коалиции
                else:
        # строим маску коалиции по наиболее "недовольным" игрокам и сохраняем эксцессы коалиций, вложенных в неё
                    main_coalition = 1 << exc_players_keys[0] # добавляем первого игрока 

                    for i in range(1, k_plus):
                        main_coalition |= 1 << exc_players_keys[i] # добавляем всех остальных

                    coalitions_keys, zero_excess_keys,coalitions = exc_subcoalitions(
                        k_plus, exc_coalitions, main_coalition, exc_players_keys) # эксцессы вложенных коалиций

        # если коалиций с положительным эксцессом нет, то строим коалицию, которая содержит в себе "недовольные" коалиций
                    if not coalitions:
                        coalitions_keys, zero_excess_keys, coalitions, exc_players_keys = find_main_coalition(
                            k_plus, exc_coalitions, exc_players_keys)

        # считаем дополнение к предрешению
                    term = sum(coalitions) / k  # общий член в формуле
                    idx = 0    # иднекс коалиций с положительным эксцессом

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

                    for player in extra_x:
                        # пересчитываем эксцессы оставшихся коалиций
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

                # пересчитываем суммы дополнений для игроков
                for player in extra_x:
                    total_extra_x[player] += extra_x[player]
            
            # пересчитываем суммы игроков
            for player in total_extra_x:
                if total_extra_x[player] > eps:
                    # ищем все коалиции, в которые входит игрок, и добавляем им дополнение к предрешению
                    for coalition in new_sum_x:
                        if coalition & (1 << player):
                            new_sum_x[coalition] += total_extra_x[player]
            old_sum_x = new_sum_x

        # сохраняем результат
        self.x = x



class CoreFinderAlt(CoreFinder):    
    def __init__(
            self,
            n: Sequence | int = None,
            v: dict = None,
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
                    idx = 0    # иднекс коалиций с положительным эксцессом

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
    

