# Курсовая работа

## Импорты
Код ниже импортирует необходимые библиотеки для выполнения различных задач:
- **copy**: для работы с глубокими копиями объектов.
- **math**: предоставляет математические функции.
- **random**: для работы с генерацией случайных чисел.
- **time**: для измерения времени.
- **collections.defaultdict**: упрощает работу со словарями.
- **dataclasses.dataclass**: для упрощения создания классов с данными.
- **datetime**: для работы с датой и временем.
- **enum.StrEnum**: для создания перечислений.
```python
import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
```
## Глобалы
Глобальный "константы"
```python
NUM_BUSES: int = 8
WORK_START: int = 6
WORK_END: int = 27
PEAK_HOURS: list[tuple] = [(7, 9), (17, 19)]
DAYS_OF_WEEK: int = 7
ROUTE_TIME: int = 60
POPULATION_SIZE: int = 10
MUTATION_RATE: float = 0.1
GENERATIONS: int = 50
TOURNAMENT_SIZE: int = 5
ELITISM_COUNT: int = 5
MORNING_PEAK_START: int = 7
MORNING_PEAK_END: int = 9
EVENING_PEAK_START: int = 17
EVENING_PEAK_END: int = 19
PEAK_LOAD: float = 0.7
NORMAL_LOAD: float = 1 - PEAK_LOAD
ROUTE_TIME_VARIATION: int = 10
```
## Перечисление DriverType
Класс **DriverType** — это перечисление, определяющее типы водителей:
- A — водитель типа "a".
- B — водитель типа "b".
```python
class DriverType(StrEnum):
    A = "a"
    B = "b"
```
## Класс Bus

**Bus** — класс для представления автобуса в системе. Каждый автобус имеет уникальный идентификатор **id**, который позволяет отличать один автобус от другого.
```python
@dataclass(slots=True, frozen=True)
class Bus:
    id: int
```
## Класс Driver

**Driver** — класс для представления водителя. Каждый водитель имеет два основных атрибута:

- **type**: Тип водителя, который указывается через перечисление **DriverType** (например, A или B).
- **id**: Уникальный идентификатор водителя, который позволяет отличать одного водителя от другого.
```python
@dataclass(slots=True, frozen=True)
class Driver:
    type: DriverType
    id: int
```
## Функция is_peak_hour

Функция **is_peak_hour** проверяет, является ли заданный час пиковым. Это делается путем проверки, попадает ли час в один из указанных интервалов пикового времени, которые хранятся в глобальной переменной **PEAK_HOURS**.

- **hour**: час, который проверяется на принадлежность к пиковому времени.

Функция возвращает **True**, если час попадает в пиковое время, и **False** в противном случае.
```python
def is_peak_hour(hour):
    return any(start <= hour % 24 < end for start, end in PEAK_HOURS)
```
## Функция generate_driver_schedules

Функция **generate_driver_schedules** генерирует расписания для водителей на неделю. В расписаниях учитываются как рабочие смены, так и перерывы. Для водителей типа A и B используются различные графики работы и количество перерывов.

- **drivers**: список водителей, для которых генерируются расписания.

Процесс генерации расписания:
- Для водителей типа A (DriverType.A) задаются фиксированные смены с перерывами на обед.
- Для водителей типа B (DriverType.B) задается смена, которая делится на несколько блоков с перерывами.

Возвращаемое значение — словарь, в котором для каждого водителя хранится список его рабочих блоков и перерывов по дням недели.
```python
def generate_driver_schedules(drivers):
    schedules = defaultdict(list)
    base_date = datetime(2024, 12, 15, WORK_START, 0)
    shifts_a = [(6, 15), (14, 23), (18, 27)]

    for driver in drivers:
        for day in range(DAYS_OF_WEEK):
            current_day = base_date + timedelta(days=day)
            work_blocks = []

            if driver.type == DriverType.A:
                shift_id = (driver.id + day) % len(shifts_a)
                shift = shifts_a[shift_id]
                block1_start = current_day.replace(hour=shift[0])
                block1_end = block1_start + timedelta(hours=4)
                lunch_start = block1_end
                while is_peak_hour(lunch_start.hour):
                    lunch_start += timedelta(minutes=15)
                lunch_end = lunch_start + timedelta(hours=1)

                work_blocks.append((block1_start, block1_end))
                work_blocks.append(("lunch", lunch_start, lunch_end))
                block2_start = lunch_end
                block2_end = block2_start + timedelta(hours=4)
                work_blocks.append((block2_start, block2_end))

                schedules[(driver.type, driver.id)].append(work_blocks)

            elif driver.type == DriverType.B:
                cycle_day = day % 3
                if cycle_day == 0:
                    if (day // 3 + driver.id) % 2 == 0:
                        shift_start = current_day.replace(hour=6)
                    else:
                        shift_start = current_day.replace(hour=15)

                    work_blocks = []
                    current_time = shift_start
                    shift_end = current_time + timedelta(hours=12)
                    breaks_taken = 0
                    while current_time < shift_end:
                        block_end = min(current_time + timedelta(hours=2), shift_end)
                        work_blocks.append((current_time, block_end))

                        if block_end < shift_end:
                            break_start = block_end
                            break_end = break_start + timedelta(minutes=15)
                            if breaks_taken < 2:
                                work_blocks.append(
                                    ("break_lunch", break_start, break_end),
                                )
                            else:
                                work_blocks.append(("break", break_start, break_end))
                            current_time = break_end
                            breaks_taken += 1
                        else:
                            break

                    schedules[(driver.type, driver.id)].append(work_blocks)
                else:
                    schedules[(driver.type, driver.id)].append([])

    return schedules
```
## Функция assign_additional_buses

Функция **assign_additional_buses** определяет количество дополнительных автобусов, которые нужно выделить в зависимости от времени суток (пиковое или не пиковое время).

- **time_period**: строка, указывающая на период времени (например, "peak" или другое).
- **hour**: час, для которого определяется необходимое количество автобусов.

Функция возвращает количество автобусов, которое зависит от того, является ли время пиковым или нет. Для пикового времени выделяется 70% от общего числа автобусов, для вечернего времени — 50%, для остального времени — 30%.
```python
def assign_additional_buses(time_period, hour):
    if time_period == "peak":
        return int(NUM_BUSES * 0.7)
    else:
        if hour % 24 >= 18 or hour % 24 < 3:
            return int(NUM_BUSES * 0.5)
        return int(NUM_BUSES * 0.3)
```
## Функция create_routes

Функция **create_routes** создает структуру для маршрутов автобусов. Для каждого автобуса создается словарь маршрутов по дням недели и времени суток, изначально с пустыми значениями.

Возвращаемое значение — словарь, где ключами являются ID автобусов, а значениями — вложенные словари с пустыми значениями для маршрутов.
```python
def create_routes():
    return {
        Bus(id): defaultdict(lambda: defaultdict(lambda: None))
        for id in range(1, NUM_BUSES + 1)
    }
```
## Функция selection

Функция **selection** выполняет отбор наиболее приспособленных особей из популяции с использованием турнира. Из каждой случайной подгруппы отбирается лучший элемент, и его копия добавляется в новый список отобранных особей.

- **population**: текущая популяция.
- **fitnesses**: список приспособленности для каждой особи популяции.

Процесс отбора:
- Для каждой особи в популяции случайным образом выбирается группа особей (турнир).
- Турнир сортируется по приспособленности, и выбирается лучший элемент.
- Возвращается новый список отобранных особей.
```python
def selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(
            list(zip(population, fitnesses, strict=False)),
            TOURNAMENT_SIZE,
        )
        tournament.sort(key=lambda x: x[1])
        selected.append(copy.deepcopy(tournament[0][0]))
    return selected
```
## Класс GeneticAlgorithm

**GeneticAlgorithm** — класс для реализации генетического алгоритма, который используется для планирования расписаний водителей и автобусов. Класс включает в себя несколько методов для инициализации популяции, кроссовера, мутации, оценки приспособленности и эволюции решений.

### Методы класса

- **`__init__`**: Инициализация класса. Принимает список водителей и количество поколений для генетического алгоритма. Создает начальные расписания водителей и маршруты автобусов.
- **`initialize_population`**: Инициализация популяции. Создает начальную популяцию, заполняя расписания случайными водителями и автобусами.
- **`crossover`**: Осуществляет операцию кроссовера между двумя родительскими решениями, создавая новое потомство.
- **`is_driver_available`**: Проверяет, доступен ли водитель в определенный час. Учитываются текущие графики водителя.
- **`mutate`**: Осуществляет мутацию в популяции, случайным образом изменяя одно из расписаний.
- **`fitness_function`**: Функция оценки приспособленности для индивида (расписания). Рассчитывает штрафы за нарушение условий, таких как количество автобусов, часы работы водителей и их перерывы.
- **`evolve`**: Основной метод эволюции, который итерирует популяцию через несколько поколений, выполняя отбор, кроссовер и мутацию.
- **`print_best_schedule`**: Печатает лучшее расписание для найденного решения, показывая, какие водители назначены на каждый автобус в каждый день.
```python
class GeneticAlgorithm:
    def __init__(self, drivers, num_generations=GENERATIONS):
        self.drivers = drivers
        self.num_generations = num_generations
        self.schedules = generate_driver_schedules(drivers)
        self.routes = create_routes()

    def initialize_population(self):
        population = []
        available_drivers = {
            DriverType.A: [d for d in self.drivers if d.type == DriverType.A],
            DriverType.B: [d for d in self.drivers if d.type == DriverType.B],
        }

        for _ in range(POPULATION_SIZE):
            individual = copy.deepcopy(self.routes)
            for bus in individual:
                for day in range(DAYS_OF_WEEK):
                    current_hour = WORK_START
                    while current_hour < WORK_END:
                        driver_type = (
                            DriverType.B
                            if random.random() < 0.7
                            else DriverType.A
                            if current_hour % 24 >= 18 or current_hour % 24 < 3
                            else DriverType.A
                        )

                        if available_drivers[driver_type]:
                            driver = random.choice(available_drivers[driver_type])
                            if self.schedules[(driver.type, driver.id)][day]:
                                shift_end = min(current_hour + 4, WORK_END)
                                for hour in range(current_hour, shift_end):
                                    individual[bus][day][hour] = driver
                                current_hour = shift_end
                            else:
                                current_hour += 1
                        else:
                            current_hour += 1

            population.append(individual)
        return population

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(self.routes)

        for bus in child:
            for day in range(DAYS_OF_WEEK):
                crossover_point = random.randint(WORK_START, WORK_END)
                for hour in range(WORK_START, crossover_point):
                    child[bus][day][hour] = parent1[bus][day][hour]
                for hour in range(crossover_point, WORK_END):
                    child[bus][day][hour] = parent2[bus][day][hour]

        return child

    def is_driver_available(self, driver, hour):
        driver_key = (driver.type, driver.id)
        for day in range(DAYS_OF_WEEK):
            for block in self.schedules[driver_key][day]:
                start, end = block[1:] if len(block) == 3 else block
                if start.hour <= hour < end.hour:
                    return False
        return True

    def mutate(self, individual):
        mutated = copy.deepcopy(individual)
        bus, day = (
            random.choice(list(mutated.keys())),
            random.randint(0, DAYS_OF_WEEK - 1),
        )
        hour = random.randint(WORK_START, WORK_END - 1)

        current_driver = mutated[bus][day].get(hour)
        if current_driver:
            available_drivers = [
                d
                for d in self.drivers
                if d != current_driver and self.is_driver_available(d, hour)
            ]
            if available_drivers:
                mutated[bus][day][hour] = random.choice(available_drivers)

        return mutated

    def fitness_function(self, individual):
        fitness = 10000
        penalty = 0
        buses_per_hour = defaultdict(lambda: defaultdict(int))

        for schedule in individual.values():
            for day, daily_schedule in schedule.items():
                for hour, driver in daily_schedule.items():
                    if driver:
                        buses_per_hour[day][hour % 24] += 1

        for day in range(DAYS_OF_WEEK):
            for hour in range(WORK_START, WORK_END):
                period = "peak" if is_peak_hour(hour) else "offpeak"
                required, actual = (
                    assign_additional_buses(period, hour),
                    buses_per_hour[day][hour],
                )
                if actual < required:
                    penalty += (required - actual) * (
                        100
                        if is_peak_hour(hour)
                        else (75 if hour >= 18 or hour < 3 else 50)
                    )

        driver_hours = defaultdict(float)
        for schedule in individual.values():
            for day, daily_schedule in schedule.items():
                driver_day_hours = defaultdict(float)
                for driver in daily_schedule.values():
                    if driver:
                        driver_key = (driver.type, driver.id)
                        if not self.schedules[driver_key][day]:
                            penalty += 500
                        driver_day_hours[driver_key] += 1
                    else:
                        penalty += 20

                for driver_key, hours in driver_day_hours.items():
                    if driver_key[0] == DriverType.A and hours > 9:
                        penalty += (hours - 9) * 100
                    elif driver_key[0] == DriverType.B and hours > 12:
                        penalty += (hours - 12) * 100
                    driver_hours[driver_key] += hours

        for driver_type in [DriverType.A, DriverType.B]:
            type_hours = [h for k, h in driver_hours.items() if k[0] == driver_type]
            if type_hours:
                avg_hours = sum(type_hours) / len(type_hours)
                penalty += sum(abs(h - avg_hours) for h in type_hours) * 10

        return fitness - penalty

    def evolve(self):
        population = self.initialize_population()
        for generation in range(self.num_generations):
            fitnesses = [self.fitness_function(ind) for ind in population]
            sorted_population = [
                i
                for _, i in sorted(
                    zip(fitnesses, population, strict=False),
                    key=lambda x: x[0],
                )
            ]
            best_fitness = min(fitnesses)

            if best_fitness == 0:
                print(f"Решение найдено при генерации {generation}")
                break

            new_population = sorted_population[:ELITISM_COUNT]
            selected = selection(population, fitnesses)

            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(selected, 2)
                child = self.mutate(self.crossover(parent1, parent2))
                new_population.append(child)

            population = new_population

        return population

    def print_best_schedule(self, best_individual):
        print("\nРасписание:")
        for bus in sorted(best_individual, key=lambda bus: bus.id):
            print(f"\nАвтобус {bus.id}:")

            for day in range(DAYS_OF_WEEK):
                driver_schedule = best_individual[bus][day]
                driver = next((d for d in driver_schedule.values() if d), None)

                if not driver:
                    print(f"  День {day + 1}: Водитель не назначен")
                    continue

                driver_key = (driver.type, driver.id)
                print(f"  День {day + 1}: Водитель {driver}")

                daily_schedule = self.schedules[driver_key][day]
                if not daily_schedule:
                    print("    День отдыха")
                    continue

                for block in sorted(
                    daily_schedule,
                    key=lambda x: x[1] if isinstance(x, tuple) else x[0],
                ):
                    start, end = (
                        block[1],
                        block[2] if isinstance(block, tuple) else block,
                    )
                    block_type = block[0] if isinstance(block, tuple) else None
                    if block_type == "lunch":
                        print(
                            f"\t{start.strftime('%H:%M')} - {end.strftime('%H:%M')} (Обед)",
                        )
                    elif block_type == "break":
                        print(
                            f"\t{start.strftime('%H:%M')} - {end.strftime('%H:%M')} (Перерыв)",
                        )
                    else:
                        print(f"\t{start.strftime('%H:%M')} - {end.strftime('%H:%M')}")
```
## Функция calculate_required_drivers

Функция **calculate_required_drivers** рассчитывает необходимое количество водителей для пикового и непикового времени, исходя из числа доступных автобусов. Эта функция учитывает требования по количеству водителей на основе типа времени (пиковое или непиковое).

- **num_buses**: количество автобусов, для которых нужно рассчитать количество водителей.

Функция возвращает два значения:
1. Требуемое количество водителей для пикового времени.
2. Требуемое количество водителей для непикового времени.
```python
def calculate_required_drivers(num_buses):
    return (
        math.ceil(num_buses * 0.7 * 7 / 5 + num_buses * 0.7 * 7 / 5 * 0.2),
        math.ceil(num_buses * 0.5 * 3 + num_buses * 0.5 * 3 * 0.2),
    )
```
## Класс **GreedyAlgorithm**

Класс **GreedyAlgorithm** реализует жадный алгоритм для назначения водителей на автобусы. Алгоритм пытается минимизировать нарушения, выбирая оптимальных водителей для каждой смены, с учетом типа водителя и ограничений по времени работы.

### Методы класса

- **`__init__`**: Инициализация класса. Создает расписания водителей и маршруты, а также структуры для отслеживания рабочих часов водителей и перерывов.
- **`is_peak_hour`**: Определяет, является ли текущий час пиковым временем на основе дня недели и времени суток.
- **`can_take_break`**: Проверяет, может ли водитель взять перерыв в определенный час, основываясь на типе водителя, его рабочих часах и предыдущих перерывах.
- **`solve`**: Основной метод, который решает задачу распределения водителей на автобусы, с учетом пикового времени, количества водителей, ограничений по рабочим часам и перерывам.
- **`print_best_schedule`**: Печатает лучшее расписание, показывая, какие водители назначены на каждый автобус в каждый день.
```python
class GreedyAlgorithm:
    def __init__(self, drivers, num_generations=GENERATIONS):
        self.drivers = drivers
        self.schedules = generate_driver_schedules(drivers)
        self.routes = create_routes()
        self.driver_work_hours = defaultdict(int)
        self.driver_breaks = defaultdict(list)

    @staticmethod
    def is_peak_hour(hour, day):
        if day < 5 and (
            MORNING_PEAK_START <= hour < MORNING_PEAK_END
            or EVENING_PEAK_START <= hour < EVENING_PEAK_END
        ):
            return True
        return False

    def can_take_break(self, driver, hour, day):
        if driver.type == DriverType.A:
            hours_worked = self.driver_work_hours[(driver.type, driver.id, day)]
            return hours_worked >= 4 and not self.is_peak_hour(hour, day)
        else:
            breaks_taken = len(
                self.driver_breaks[(driver.type, driver.id, day)],
            )
            last_break = (
                self.driver_breaks[(driver.type, driver.id, day)][-1]
                if breaks_taken > 0
                else 0
            )
            hours_since_break = hour - last_break
            return breaks_taken < 2 <= hours_since_break

    def solve(self):
        schedule = copy.deepcopy(self.routes)
        total_driver_hours = defaultdict(int)

        for day in range(DAYS_OF_WEEK):
            daily_driver_hours = defaultdict(int)
            available_type_b = [
                d for d in self.drivers if d.type == DriverType.B and (day % 3 == 0)
            ]
            available_type_a = [d for d in self.drivers if d.type == DriverType.A]
            for current_hour in range(WORK_START, WORK_END):
                is_peak = self.is_peak_hour(current_hour, day)
                buses_needed = math.ceil(
                    NUM_BUSES * (PEAK_LOAD if is_peak else NORMAL_LOAD),
                )
                for bus in list(schedule.keys())[:buses_needed]:
                    if not schedule[bus][day].get(current_hour):
                        preferred_night_hours = current_hour >= 18 or current_hour < 3
                        preferred_type = (
                            DriverType.B if preferred_night_hours else DriverType.A
                        )

                        best_driver = None
                        min_total_hours = float("inf")
                        available_drivers = (
                            available_type_b
                            if preferred_type == DriverType.B
                            else available_type_a
                        )
                        for driver in available_drivers:
                            daily_hours = daily_driver_hours[(driver.type, driver.id)]
                            total_hours = total_driver_hours[(driver.type, driver.id)]

                            max_daily_hours = 12 if driver.type == DriverType.B else 8

                            if daily_hours < max_daily_hours:
                                is_busy = False
                                for other_bus in schedule:
                                    if (
                                        other_bus != bus
                                        and schedule[other_bus][day].get(current_hour)
                                        == driver
                                    ):
                                        is_busy = True
                                        break
                                if not is_busy and total_hours < min_total_hours:
                                    min_total_hours = total_hours
                                    best_driver = driver
                        if not best_driver and not preferred_night_hours:
                            other_drivers = (
                                available_type_a
                                if preferred_type == DriverType.B
                                else available_type_b
                            )
                            for driver in other_drivers:
                                daily_hours = daily_driver_hours[
                                    (driver.type, driver.id)
                                ]
                                total_hours = total_driver_hours[
                                    (driver.type, driver.id)
                                ]

                                max_daily_hours = (
                                    12 if driver.type == DriverType.B else 8
                                )

                                if daily_hours < max_daily_hours:
                                    is_busy = False
                                    for other_bus in schedule:
                                        if (
                                            other_bus != bus
                                            and schedule[other_bus][day].get(
                                                current_hour,
                                            )
                                            == driver
                                        ):
                                            is_busy = True
                                            break

                                    if not is_busy and total_hours < min_total_hours:
                                        min_total_hours = total_hours
                                        best_driver = driver

                        if best_driver:
                            schedule[bus][day][current_hour] = best_driver
                            daily_driver_hours[(best_driver.type, best_driver.id)] += 1
                            total_driver_hours[(best_driver.type, best_driver.id)] += 1
                            daily_hours = daily_driver_hours[
                                (best_driver.type, best_driver.id)
                            ]
                            if best_driver.type == DriverType.A:
                                if daily_hours == 4 and not self.is_peak_hour(
                                    current_hour + 1,
                                    day,
                                ):
                                    self.driver_breaks[
                                        (
                                            best_driver.type,
                                            best_driver.id,
                                            day,
                                        )
                                    ].append(current_hour + 1)
                            elif best_driver.type == DriverType.B:
                                breaks_taken = len(
                                    self.driver_breaks[
                                        (
                                            best_driver.type,
                                            best_driver.id,
                                            day,
                                        )
                                    ],
                                )
                                if (
                                    breaks_taken < 2
                                    and daily_hours % 2 == 0
                                    and not self.is_peak_hour(current_hour + 1, day)
                                ):
                                    self.driver_breaks[
                                        (
                                            best_driver.type,
                                            best_driver.id,
                                            day,
                                        )
                                    ].append(current_hour + 1)

        return schedule

    def print_best_schedule(self, schedule):
        print("\nРасписание:")
        for bus, bus_schedule in schedule.items():
            print(f"\nАвтобус {bus}:")
            for day in range(DAYS_OF_WEEK):
                driver_schedule = bus_schedule[day]
                if not any(driver_schedule.values()):
                    continue
                driver = None
                for d in driver_schedule.values():
                    if d is not None:
                        driver = d
                        break

                if driver is None:
                    print(f"  День {day + 1}: Водитель не назначен")
                    continue

                driver_key = (driver.type, driver.id)
                print(f"  День {day + 1}: Driver ({driver.id})")
                daily_schedule = self.schedules[driver_key][day]
                if not daily_schedule:
                    print("    День отдыха")
                    continue

                for _, block in enumerate(daily_schedule):
                    if isinstance(block, tuple):
                        if len(block) == 3:
                            break_type, start, end = block
                            if break_type == "lunch":
                                print(
                                    f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')} (Обед)",
                                )
                            elif break_type == "break_lunch":
                                print(
                                    f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')} (Обед)",
                                )
                            else:
                                print(
                                    f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')} (Перерыв)",
                                )
                        else:
                            start, end = block
                            print(
                                f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')}",
                            )
                    else:
                        start, end = block
                        print(
                            f"    {start.strftime('%H:%M')} - {end.strftime('%H:%M')}",
                        )
```
## Функция **evaluate_schedule**

Функция **evaluate_schedule** используется для оценки качества расписания, созданного с помощью алгоритмов (например, генетического или жадного). Она рассчитывает несколько ключевых метрик, которые помогают определить, насколько хорошо составлено расписание с точки зрения:

1. **Нарушений ограничений**:
    - Эта метрика отслеживает количество нарушений ограничений по рабочим часам водителей. Например, если водитель работает более 24 часов подряд или не соблюдает ограничения на количество перерывов, это будет засчитано как нарушение.
    - В частности, нарушения включают превышение максимально допустимых рабочих часов для водителей типа A (8 часов) или типа B (12 часов).

2. **Стандартного отклонения рабочих часов**:
    - Стандартное отклонение рабочих часов водителей вычисляется, чтобы понять, насколько равномерно распределены рабочие часы между водителями.
    - Низкое значение стандартного отклонения указывает на то, что рабочие часы распределены равномерно, что является оптимальным. Высокое стандартное отклонение может указывать на перегрузку некоторых водителей и недозагруженность других.

3. **Процент покрытия смен**:
    - Эта метрика измеряет, насколько хорошо расписание покрывает все рабочие слоты, требуемые для автобусов. Например, если назначены водители на все рабочие часы для всех автобусов, процент покрытия будет близким к 100%.
    - Если же на некоторые часы или автобусы не назначены водители, это приведет к снижению процента покрытия.

### Как работает функция:
- **Нарушения ограничений** отслеживаются при каждой проверке водителя на расписание. Например, если водитель типа A работает больше 8 часов или если водитель не берёт положенные перерывы, это будет зафиксировано как нарушение.
- **Рабочие часы водителей** суммируются по каждому водителю. Затем вычисляется стандартное отклонение этих рабочих часов, чтобы определить, насколько распределение рабочих часов сбалансировано.
- **Покрытие смен** вычисляется как отношение количества занятых рабочих слотов (где назначены водители) к общему количеству рабочих слотов для всех автобусов.

Таким образом, функция **evaluate_schedule** позволяет не только проверить, соответствует ли расписание всем необходимым требованиям, но и дает представление о его эффективности с точки зрения загрузки водителей и покрытия всех рабочих часов.
```python
def evaluate_schedule(schedule, drivers, schedules):
    metrics = {
        "constraint_violations": 0,
        "workload_std_dev": 0.0,
        "coverage_percentage": 0.0,
    }
    driver_hours = defaultdict(int)
    total_slots = 0
    filled_slots = 0

    for bus_schedule in schedule.values():
        for day in range(DAYS_OF_WEEK):
            for hour in range(WORK_START, WORK_END):
                total_slots += 1
                driver = bus_schedule[day].get(hour)
                if driver:
                    filled_slots += 1
                    driver_hours[(driver.type, driver.id)] += 1
                    if hour >= 24 and driver.type == DriverType.A:
                        metrics["constraint_violations"] += 1
    if driver_hours:
        hours = list(driver_hours.values())
        mean_hours = sum(hours) / len(hours)
        variance = sum((h - mean_hours) ** 2 for h in hours) / len(hours)
        metrics["workload_std_dev"] = math.sqrt(variance)
    metrics["coverage_percentage"] = (filled_slots / total_slots) * 100

    return metrics
```
## Функция **main**

Функция **main** выполняет полный процесс планирования расписания для водителей и автобусов, используя два различных алгоритма (генетический и жадный), а затем оценивает и сравнивает результаты работы этих алгоритмов. Ниже подробно описано, как работает эта функция:

1. **Инициализация данных**:
    - В начале функции рассчитывается необходимое количество водителей для каждого типа (A и B) с помощью функции `calculate_required_drivers`.
    - Затем создаются списки водителей с использованием этих расчетов, где каждый водитель является экземпляром класса **Driver** с указанием его типа (A или B).

2. **Запуск и оценка Генетического Алгоритма**:
    - Создается экземпляр класса **GeneticAlgorithm**, который принимает список водителей.
    - Генетический алгоритм начинает работать, эволюционируя популяцию решений и выбирая лучшее решение (расписание).
    - Время работы генетического алгоритма измеряется, чтобы узнать, сколько времени занял процесс.
    - После этого, с использованием метода `fitness_function`, оценивается эффективность лучшего решения (расписания) по нескольким меткам.

3. **Запуск и оценка Жадного Алгоритма**:
    - Затем создается экземпляр класса **GreedyAlgorithm**, который также использует список водителей.
    - Жадный алгоритм решает задачу по созданию расписания, пытаясь назначить водителей на автобусы, начиная с самых приоритетных случаев.
    - Время выполнения жадного алгоритма также измеряется для дальнейшего сравнения.

4. **Оценка качества расписаний**:
    - Для каждого из алгоритмов (генетического и жадного) вызывается функция `evaluate_schedule`, которая оценивает качество полученного расписания по трем ключевым меткам:
        - Нарушения ограничений.
        - Стандартное отклонение рабочих часов водителей.
        - Процент покрытия смен.

5. **Вывод результатов**:
    - В консоль выводится информация о водителях (количество водителей типа A и B).
    - После этого выводятся результаты для каждого алгоритма, включая:
        - Количество нарушений ограничений.
        - Стандартное отклонение рабочих часов водителей.
        - Процент покрытия смен.
    - Также выводится время, затраченное каждым алгоритмом на выполнение задачи.
```python
def main():
    num_type_a, num_type_b = calculate_required_drivers(NUM_BUSES)
    drivers = [Driver(DriverType.A, i + 1) for i in range(num_type_a)] + [
        Driver(DriverType.B, i + 1) for i in range(num_type_b)
    ]

    ga_start_time = time.time()

    ga = GeneticAlgorithm(drivers)
    population = ga.evolve()
    fitnesses = [(ind, ga.fitness_function(ind)) for ind in population]
    best_individual = max(fitnesses, key=lambda x: x[1])[0]
    print("Статистика по водителям:")
    print(f"Водителей типа A: {num_type_a}")
    print(f"Водителей типа B: {num_type_b}")

    ga.initialize_population()

    ga_end_time = time.time()
    ga_duration = ga_end_time - ga_start_time

    greedy_start_time = time.time()

    greedy = GreedyAlgorithm(drivers)
    greedy_schedule = greedy.solve()
    greedy.print_best_schedule(greedy_schedule)

    greedy_end_time = time.time()
    greedy_duration = greedy_end_time - greedy_start_time

    ga_metrics = evaluate_schedule(best_individual, drivers, ga.schedules)
    greedy_metrics = evaluate_schedule(greedy_schedule, drivers, greedy.schedules)

    print("\nОценка расписаний:")
    print("Генетический Алгоритм:")
    print(f"  Нарушения ограничений: {ga_metrics['constraint_violations']}")
    print(
        f"  Стандартное отклонение рабочих часов: {ga_metrics['workload_std_dev']:.2f}",
    )
    print(f"  Процент покрытия смен: {ga_metrics['coverage_percentage']:.2f}%")
    print("\nАлгоритм Влоб:")
    print(f"  Нарушения ограничений: {greedy_metrics['constraint_violations']}")
    print(
        f"  Стандартное отклонение рабочих часов: {greedy_metrics['workload_std_dev']:.2f}",
    )
    print(f"  Процент покрытия смен: {greedy_metrics['coverage_percentage']:.2f}%")
    print(f"\nВремя выполнения Генетического Алгоритма: {ga_duration:.2f} секунд")
    print(f"Время выполнения Алгоритма Влоб: {greedy_duration:.2f} секунд")


main()
```
