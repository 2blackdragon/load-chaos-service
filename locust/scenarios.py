"""
scenarios.py — пул сценариев и мини-паттернов для ScenarioShape
Версия 2.0: 14-дневный прогон, нелинейная модель ошибок, аномалии, недельная сезонность.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
НОВЫЕ ВОЗМОЖНОСТИ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. WeeklyEnvelope — масштабирует нагрузку по дням недели и росту тренда.
   Используется ScenarioShape для умножения users/spawn_rate каждого тика.

2. ErrorModel — нелинейная зависимость error_rate от нагрузки:
       base = sigmoid((users − saturation_threshold) / steepness)
   Плюс случайные инциденты (Пуассон) и мультипликатор для degraded-паттернов.
   Итог: ошибки перестают быть константой и коррелируют с нагрузкой.

3. AnomalyInjector — редкие, реалистичные аномалии с метками для ML:
       MEMORY_LEAK   — медленный drift users вверх
       SLOW_DEPLOY   — резкий рост latency на короткий период
       DB_SATURATION — error spike при длительной высокой нагрузке
       TRAFFIC_BURST — внезапный +30–80% users на несколько минут
   Каждая аномалия несёт метку severity (low/medium/high/critical).

4. FourteenDayPlan — раскрывает 14-дневный план прогонов,
   учитывая день недели и вероятность пропуска стейджей.
   Возвращает список DayRun (день → список стейджей + параметры огибающей).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ИНТЕГРАЦИЯ В ScenarioShape
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    plan = FourteenDayPlan()
    for day_run in plan:
        stages = day_run.stages           # list[dict] — обычный список стейджей
        envelope = day_run.envelope       # WeeklyEnvelope — коэффициент масштаба
        for stage in stages:
            users = stage["users"] * envelope.scale_factor
            error_rate = ERROR_MODEL.compute(users, stage.get("mini_pattern"))
            anomaly = ANOMALY_INJECTOR.tick(users, stage)
            # метрики для ML:
            # users, spawn_rate, error_rate, latency_ms, anomaly.label, anomaly.severity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ПОЧЕМУ ТАК РАБОТАЕТ ДЛЯ ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Для обучения модели прогнозирования нагрузки и сбоев важно:
- ошибки → нелинейная корреляция с users (sigmoid + Poisson burst)
- аномалии → редкие, но помеченные (imbalanced-friendly для классификации)
- дрейф → тренд за 14 дней (для seq2seq и LSTM)
- сезонность → час суток + день недели (для SARIMA / Temporal Fusion)
- шум → у каждого мини-паттерна свой noise, нет идеальных прямых линий
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# build_stages — раскрытие сценария с вероятностным пропуском стейджей
# ═══════════════════════════════════════════════════════════════════════

def build_stages(scenario_name: str) -> list[dict]:
    """Вернуть список стейджей для прогона, учитывая skip_probability.

    Для каждого стейджа с ключом ``skip_probability`` бросается монета:
    если случайное число < skip_probability — стейдж пропускается.
    Стейджи без этого ключа (обязательные) включаются всегда.

    При пропуске стейджа transition следующего за ним стейджа
    автоматически увеличивается вдвое, чтобы переход оставался плавным.

    Args:
        scenario_name: ключ из словаря SCENARIOS.

    Returns:
        Список стейджей (копии словарей, оригиналы не изменяются).

    Raises:
        KeyError: если сценарий не найден.
    """
    raw = SCENARIOS[scenario_name]
    result: list[dict] = []
    prev_was_skipped = False

    for stage in raw:
        p_skip = stage.get("skip_probability", 0.0)
        if p_skip > 0.0 and random.random() < p_skip:
            log.debug("stage '%s' skipped (p=%.2f)", stage["name"], p_skip)
            prev_was_skipped = True
            continue

        s = stage.copy()

        duration_min = s.get("duration_min", 0)
        duration_max = s.get("duration_max", duration_min)
        s["_duration_resolved"] = random.randint(duration_min, duration_max)

        if prev_was_skipped and "transition" in s:
            s["transition"] = s["transition"] * 2
            log.debug(
                "stage '%s': transition doubled to %ds (prev stage was skipped)",
                s["name"], s["transition"],
            )
        result.append(s)
        prev_was_skipped = False

    return result


# ═══════════════════════════════════════════════════════════════════════
# Мини-паттерны
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MiniPattern:
    """Короткая вставка (30–360 сек) с конкретным характером нагрузки."""

    name: str
    duration_min: int       # минимальная длительность, секунд
    duration_max: int       # максимальная длительность, секунд
    users_min: int
    users_max: int
    spawn_min: float
    spawn_max: float
    noise: float = 0.04     # ±% шума на каждом тике (меньше = плавнее)
    weight: float = 1.0     # базовый вес для взвешенного выбора
    # Мультипликатор к error_rate для аномальных паттернов.
    # 1.0 — норма, >1 — больше ошибок (degraded ×3, pulse ×1.8 и т.д.)
    error_multiplier: float = 1.0
    # Мультипликатор к latency_ms
    latency_multiplier: float = 1.0

    @property
    def duration(self) -> int:
        """Случайная длительность из диапазона (вызывается при старте паттерна)."""
        return random.randint(self.duration_min, self.duration_max)


MINI_PATTERNS: dict[str, MiniPattern] = {
    # Тихий провал — как будто все ушли на обед
    "lull": MiniPattern(
        name="lull",
        duration_min=90, duration_max=150,
        users_min=5, users_max=12,
        spawn_min=0.5, spawn_max=1.2,
        noise=0.03,
        error_multiplier=0.4,   # в тишине ошибок почти нет
        latency_multiplier=0.7,
    ),
    # Микро-спайк — короткий резкий всплеск
    "micro_spike": MiniPattern(
        name="micro_spike",
        duration_min=45, duration_max=90,
        users_min=70, users_max=130,
        spawn_min=25.0, spawn_max=50.0,
        noise=0.08,
        error_multiplier=2.5,   # спайк → временное превышение порога
        latency_multiplier=1.9,
    ),
    # Волна — плавный рост и спад
    "wave": MiniPattern(
        name="wave",
        duration_min=150, duration_max=240,
        users_min=30, users_max=100,
        spawn_min=5.0, spawn_max=12.0,
        noise=0.03,
        error_multiplier=1.0,
        latency_multiplier=1.1,
    ),
    # Пульс — несколько быстрых всплесков подряд
    "pulse": MiniPattern(
        name="pulse",
        duration_min=75, duration_max=120,
        users_min=150, users_max=190,
        spawn_min=18.0, spawn_max=45.0,
        noise=0.10,
        error_multiplier=1.8,
        latency_multiplier=1.6,
    ),
    # Агония — высокая нагрузка с умеренным шумом (degraded period)
    "degraded": MiniPattern(
        name="degraded",
        duration_min=120, duration_max=180,
        users_min=180, users_max=350,
        spawn_min=35.0, spawn_max=70.0,
        noise=0.12,
        error_multiplier=3.5,   # главный источник высоких ошибок
        latency_multiplier=2.8,
    ),
    # Восстановление — плавный спад после spike
    "recovery": MiniPattern(
        name="recovery",
        duration_min=100, duration_max=150,
        users_min=10, users_max=35,
        spawn_min=2.0, spawn_max=4.5,
        noise=0.03,
        error_multiplier=0.6,
        latency_multiplier=0.85,
    ),
    # Ночное дежурство — совсем тихо, один-два пользователя
    "night_watch": MiniPattern(
        name="night_watch",
        duration_min=240, duration_max=360,
        users_min=2, users_max=7,
        spawn_min=0.3, spawn_max=0.9,
        noise=0.02,
        error_multiplier=0.2,
        latency_multiplier=0.6,
    ),
    # Batch-шторм — ровная высокая нагрузка без шума (ночные расчёты)
    "batch_storm": MiniPattern(
        name="batch_storm",
        duration_min=200, duration_max=280,
        users_min=7, users_max=12,
        spawn_min=3.0, spawn_max=5.0,
        noise=0.02,
        error_multiplier=1.3,   # batch → умеренный рост ошибок БД
        latency_multiplier=1.4,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# ErrorModel — нелинейная зависимость ошибок от нагрузки
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ErrorModel:
    """Вычисляет мгновенный error_rate для текущего тика.

    Формула:
        base = sigmoid((users − saturation_threshold) / steepness)
        incident = Bernoulli(p_incident_per_tick) * incident_magnitude
        pattern_mult = мультипликатор от текущего мини-паттерна
        error_rate = clamp(base * pattern_mult + incident, 0, max_error_rate)

    Смысл параметров:
        saturation_threshold — количество пользователей, при котором система
            начинает давать заметное число ошибок (≈ «колено» кривой).
        steepness — насколько резко растут ошибки вокруг порога:
            меньше → более резкий S-образный переход.
        p_incident_per_tick — вероятность случайного инцидента на тик
            (Пуассон с малым λ → редкие всплески).
        incident_magnitude_min/max — диапазон дополнительных ошибок при инциденте.

    Почему это важно для ML:
        - Ошибки нелинейно коррелируют с нагрузкой → модель видит реальный signal.
        - Случайные инциденты создают редкие аномальные пики → imbalanced dataset.
        - pattern_mult делает ошибки зависимыми от типа паттерна → добавляет
          ещё одну переменную для feature engineering.
    """

    saturation_threshold: float = 120.0
    steepness: float = 30.0
    base_floor: float = 0.002       # минимальный фоновый error rate
    base_ceiling: float = 0.35      # максимальный error rate системы
    p_incident_per_tick: float = 0.008  # ~0.8% вероятность инцидента на тик
    incident_magnitude_min: float = 0.05
    incident_magnitude_max: float = 0.25
    incident_duration_ticks: int = 0    # внутренний счётчик длительности инцидента
    _current_incident: float = 0.0      # текущая добавка от инцидента

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def compute(
        self,
        users: float,
        mini_pattern_name: Optional[str] = None,
        anomaly_multiplier: float = 1.0,
    ) -> float:
        """Вернуть error_rate [0..1] для текущего тика.

        Args:
            users: текущее количество пользователей.
            mini_pattern_name: имя активного мини-паттерна (для мультипликатора).
            anomaly_multiplier: внешний множитель от AnomalyInjector.

        Returns:
            Мгновенный error_rate в диапазоне [0, base_ceiling].
        """
        # Базовый sigmoid
        normalized = (users - self.saturation_threshold) / self.steepness
        base = self._sigmoid(normalized) * 0.30 + self.base_floor

        # Мультипликатор мини-паттерна
        pattern_mult = 1.0
        if mini_pattern_name and mini_pattern_name in MINI_PATTERNS:
            pattern_mult = MINI_PATTERNS[mini_pattern_name].error_multiplier

        # Инцидент: уменьшаем счётчик или бросаем новый
        if self.incident_duration_ticks > 0:
            self.incident_duration_ticks -= 1
        else:
            self._current_incident = 0.0
            if random.random() < self.p_incident_per_tick:
                self._current_incident = random.uniform(
                    self.incident_magnitude_min,
                    self.incident_magnitude_max,
                )
                # Инцидент длится 3–12 тиков (~30–120 сек при тике 10 сек)
                self.incident_duration_ticks = random.randint(3, 12)
                log.debug(
                    "incident: +%.3f error for %d ticks",
                    self._current_incident, self.incident_duration_ticks,
                )

        rate = base * pattern_mult * anomaly_multiplier + self._current_incident
        return max(self.base_floor, min(rate, self.base_ceiling))


ERROR_MODEL = ErrorModel()


# ═══════════════════════════════════════════════════════════════════════
# WeeklyEnvelope — масштабирование нагрузки по дням недели
# ═══════════════════════════════════════════════════════════════════════

# Коэффициент нагрузки по дням недели (0=пн, 6=вс).
# Рабочие дни чуть отличаются: пн медленный старт, пт — пик.
_WEEKDAY_FACTORS = {
    0: 0.85,  # понедельник — медленный старт
    1: 1.00,  # вторник
    2: 1.05,  # среда — чаще всего пик
    3: 1.00,  # четверг
    4: 0.95,  # пятница — немного спокойнее к вечеру
    5: 0.60,  # суббота
    6: 0.40,  # воскресенье
}


@dataclass
class WeeklyEnvelope:
    """Масштабирующий коэффициент для одного дня 14-дневного прогона.

    Учитывает:
    - день недели (weekday_factor)
    - линейный тренд роста нагрузки за 14 дней (growth_trend)
    - небольшой случайный jitter ±10% для реализма

    Применение:
        effective_users = stage_users * envelope.scale_factor
        effective_spawn  = stage_spawn  * envelope.scale_factor
    """

    day_index: int              # 0–13 (порядковый номер дня прогона)
    weekday: int                # 0=пн, 6=вс
    growth_rate_per_day: float = 0.011  # +1.1% нагрузки в день (≈+15% за 14 дней)
    jitter_pct: float = 0.10            # ±10% случайного шума на день

    @property
    def scale_factor(self) -> float:
        """Итоговый множитель для текущего дня."""
        weekday_f = _WEEKDAY_FACTORS.get(self.weekday, 1.0)
        trend_f = 1.0 + self.growth_rate_per_day * self.day_index
        jitter_f = 1.0 + random.uniform(-self.jitter_pct, self.jitter_pct)
        return weekday_f * trend_f * jitter_f

    @property
    def is_weekend(self) -> bool:
        return self.weekday >= 5

    @property
    def label(self) -> str:
        """Человекочитаемая метка для логов и CSV."""
        names = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
        return f"день {self.day_index + 1} ({names[self.weekday]})"


# ═══════════════════════════════════════════════════════════════════════
# AnomalyInjector — редкие реалистичные аномалии с метками для ML
# ═══════════════════════════════════════════════════════════════════════

class AnomalySeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    NONE = "none"
    MEMORY_LEAK = "memory_leak"     # медленный drift users
    SLOW_DEPLOY = "slow_deploy"     # latency spike
    DB_SATURATION = "db_saturation" # error spike при долгой высокой нагрузке
    TRAFFIC_BURST = "traffic_burst" # внезапный +50% users


@dataclass
class AnomalyEvent:
    """Результат одного тика AnomalyInjector."""

    anomaly_type: AnomalyType = AnomalyType.NONE
    severity: AnomalySeverity = AnomalySeverity.NONE
    users_multiplier: float = 1.0   # множитель к users (memory_leak, traffic_burst)
    latency_multiplier: float = 1.0 # множитель к latency_ms (slow_deploy)
    error_multiplier: float = 1.0   # множитель к error_rate (db_saturation)
    is_anomaly: bool = False         # бинарная метка для ML

    @property
    def label(self) -> str:
        """Метка для датасета (строка)."""
        return self.anomaly_type.value


@dataclass
class AnomalyInjector:
    """Генерирует редкие аномалии в процессе прогона.

    Принцип:
        Каждый тик проверяется вероятность старта новой аномалии.
        Аномалия длится несколько тиков, возвращая AnomalyEvent с множителями.
        После завершения система «восстанавливается» (multiplier → 1.0).

    Используется для:
        - создания реалистичных аномальных периодов в датасете
        - разметки данных (is_anomaly=True) для задачи классификации
        - тестирования детектора аномалий на ML-модели

    Параметры вероятностей подобраны так, чтобы аномалии составляли
    ≈3–7% тиков датасета (типичное соотношение для имбалансных задач).
    """

    # Вероятность старта каждого типа аномалии на тик
    p_memory_leak: float = 0.0005       # ~1 раз в 30 мин при тике 10 сек
    p_slow_deploy: float = 0.0003
    p_db_saturation: float = 0.0004
    p_traffic_burst: float = 0.0006

    # Внутреннее состояние
    _active: Optional[AnomalyType] = field(default=None, repr=False)
    _ticks_remaining: int = field(default=0, repr=False)
    _current_event: AnomalyEvent = field(
        default_factory=AnomalyEvent, repr=False
    )
    _high_load_ticks: int = field(default=0, repr=False)  # для db_saturation

    def tick(self, users: float, stage: dict) -> AnomalyEvent:
        """Обновить состояние на один тик и вернуть AnomalyEvent.

        Args:
            users: текущее число пользователей (после применения envelope).
            stage: текущий стейдж (для контекстных условий).

        Returns:
            AnomalyEvent с множителями и меткой.
        """
        # Отслеживаем длительность высокой нагрузки для db_saturation
        if users > 200:
            self._high_load_ticks += 1
        else:
            self._high_load_ticks = max(0, self._high_load_ticks - 2)

        # Продолжаем текущую аномалию
        if self._active is not None and self._ticks_remaining > 0:
            self._ticks_remaining -= 1
            if self._ticks_remaining == 0:
                log.debug("anomaly '%s' ended", self._active.value)
                self._active = None
                self._current_event = AnomalyEvent()
            return self._current_event

        # Пробуем запустить новую аномалию
        event = self._try_start(users, stage)
        self._current_event = event
        return event

    def _try_start(self, users: float, stage: dict) -> AnomalyEvent:
        """Случайно запустить одну аномалию или вернуть пустой event."""
        # DB_SATURATION — только при длительной высокой нагрузке
        if (
            self._high_load_ticks > 120  # >20 мин при тике 10 сек
            and random.random() < self.p_db_saturation * 5
        ):
            return self._start(
                AnomalyType.DB_SATURATION,
                ticks=random.randint(18, 60),   # 3–10 мин
                severity=AnomalySeverity.HIGH,
                error_multiplier=random.uniform(4.0, 8.0),
                latency_multiplier=random.uniform(2.5, 4.0),
            )

        # Memory leak — предпочтительно в stress-стейджах
        if random.random() < self.p_memory_leak:
            stage_name = stage.get("name", "")
            mult_boost = 1.5 if "stress" in stage_name else 1.0
            return self._start(
                AnomalyType.MEMORY_LEAK,
                ticks=random.randint(60, 180),   # 10–30 мин
                severity=AnomalySeverity.MEDIUM,
                # users_multiplier растёт линейно — имитирует медленный leak
                users_multiplier=random.uniform(1.05, 1.20) * mult_boost,
                error_multiplier=random.uniform(1.2, 2.0),
            )

        # Slow deploy — в любом стейдже
        if random.random() < self.p_slow_deploy:
            return self._start(
                AnomalyType.SLOW_DEPLOY,
                ticks=random.randint(9, 30),    # 1.5–5 мин
                severity=AnomalySeverity.LOW,
                latency_multiplier=random.uniform(2.0, 4.5),
                error_multiplier=random.uniform(1.1, 1.8),
            )

        # Traffic burst — внезапный кратковременный приток
        if random.random() < self.p_traffic_burst:
            return self._start(
                AnomalyType.TRAFFIC_BURST,
                ticks=random.randint(6, 18),    # 1–3 мин
                severity=AnomalySeverity.MEDIUM,
                users_multiplier=random.uniform(1.3, 1.8),
                error_multiplier=random.uniform(1.5, 3.0),
                latency_multiplier=random.uniform(1.3, 2.0),
            )

        return AnomalyEvent()

    def _start(
        self,
        anomaly_type: AnomalyType,
        ticks: int,
        severity: AnomalySeverity,
        users_multiplier: float = 1.0,
        latency_multiplier: float = 1.0,
        error_multiplier: float = 1.0,
    ) -> AnomalyEvent:
        self._active = anomaly_type
        self._ticks_remaining = ticks
        log.info(
            "anomaly START: %s (severity=%s, ticks=%d)",
            anomaly_type.value, severity.value, ticks,
        )
        return AnomalyEvent(
            anomaly_type=anomaly_type,
            severity=severity,
            users_multiplier=users_multiplier,
            latency_multiplier=latency_multiplier,
            error_multiplier=error_multiplier,
            is_anomaly=True,
        )


ANOMALY_INJECTOR = AnomalyInjector()


# ═══════════════════════════════════════════════════════════════════════
# FourteenDayPlan — раскрытие 14-дневного плана прогонов
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DayRun:
    """Один день 14-дневного прогона."""

    day_index: int                    # 0–13
    weekday: int                      # 0=пн, 6=вс
    scenario_name: str                # ключ из SCENARIOS
    stages: list[dict]                # раскрытый список стейджей
    envelope: WeeklyEnvelope          # коэффициент масштаба нагрузки

    @property
    def total_duration_min(self) -> int:
        """Минимальная суммарная длительность стейджей дня, секунд."""
        return sum(s.get("duration_min", 0) for s in self.stages)

    @property
    def total_duration_max(self) -> int:
        """Максимальная суммарная длительность стейджей дня, секунд."""
        return sum(s.get("duration_max", 0) for s in self.stages)


def build_fourteen_day_plan(
    start_weekday: int = 0,     # 0=пн — с какого дня начинаем прогон
    scenario_name: str = "dataset_day",
) -> list[DayRun]:
    """Построить 14-дневный план прогонов.

    Каждый день генерируется независимо:
        - свой seed пропусков стейджей (build_stages)
        - своя WeeklyEnvelope (день недели + тренд)

    В выходные (сб, вс) используется облегчённый сценарий:
        - skip_probability всех необязательных стейджей удваивается
        - или напрямую используется quick_trial как шаблон структуры

    Args:
        start_weekday: день недели первого дня (0=пн, …, 6=вс).
        scenario_name: базовый сценарий (обычно "dataset_day").

    Returns:
        Список из 14 DayRun.
    """
    plan: list[DayRun] = []

    for day_index in range(14):
        weekday = (start_weekday + day_index) % 7
        is_weekend = weekday >= 5

        # В выходные применяем более агрессивный skip для необязательных стейджей
        if is_weekend:
            stages = _build_stages_weekend(scenario_name)
        else:
            stages = build_stages(scenario_name)

        envelope = WeeklyEnvelope(
            day_index=day_index,
            weekday=weekday,
        )

        plan.append(DayRun(
            day_index=day_index,
            weekday=weekday,
            scenario_name=scenario_name,
            stages=stages,
            envelope=envelope,
        ))

        log.info(
            "%s: %d стейджей, scale=%.2f, duration≈%dч–%dч",
            envelope.label,
            len(stages),
            envelope.scale_factor,
            plan[-1].total_duration_min // 3600,
            plan[-1].total_duration_max // 3600,
        )

    return plan


def _build_stages_weekend(scenario_name: str) -> list[dict]:
    """Раскрыть сценарий с удвоенными skip_probability для выходного дня."""
    raw = SCENARIOS[scenario_name]
    result: list[dict] = []
    prev_was_skipped = False

    for stage in raw:
        p_skip = stage.get("skip_probability", 0.0)
        # В выходные: удваиваем вероятность пропуска, но не выше 0.90
        p_skip_weekend = min(p_skip * 2.0, 0.90) if p_skip > 0 else 0.0

        if p_skip_weekend > 0.0 and random.random() < p_skip_weekend:
            log.debug(
                "WEEKEND: stage '%s' skipped (p=%.2f→%.2f)",
                stage["name"], p_skip, p_skip_weekend,
            )
            prev_was_skipped = True
            continue

        s = stage.copy()

        duration_min = s.get("duration_min", 0)
        duration_max = s.get("duration_max", duration_min)
        s["_duration_resolved"] = random.randint(duration_min, duration_max)

        if prev_was_skipped and "transition" in s:
            s["transition"] = s["transition"] * 2
        result.append(s)
        prev_was_skipped = False

    return result


# ═══════════════════════════════════════════════════════════════════════
# MetricsTick — снимок метрик за один тик (удобно для CSV/Parquet)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetricsTick:
    """Снимок всех метрик для одного тика прогона.

    Используется ScenarioShape для формирования датасета.
    Каждая строка в CSV/Parquet соответствует одному MetricsTick.

    Поля, полезные для ML:
        Фичи:
            timestamp, day_index, weekday, hour, stage_name,
            mini_pattern, users, spawn_rate, scale_factor,
            is_weekend, is_anomaly, anomaly_type

        Таргеты (то, что предсказывает модель):
            error_rate     — регрессия или классификация порогов
            latency_ms     — регрессия задержки
            anomaly_type   — многоклассовая классификация
            severity       — ординальная классификация

        Полезные лаги (для seq2seq / LSTM):
            создайте скользящее окно из N последних MetricsTick
    """

    timestamp: float            # unix timestamp
    day_index: int              # 0–13
    weekday: int                # 0=пн, 6=вс
    hour: float                 # час суток (дробный, 0–24)
    stage_name: str
    mini_pattern: str           # имя активного мини-паттерна или ""
    users: float                # реальное число пользователей после envelope
    spawn_rate: float
    scale_factor: float         # коэффициент WeeklyEnvelope
    error_rate: float           # [0, 0.35]
    latency_ms: float           # задержка, мс
    is_weekend: bool
    is_anomaly: bool            # бинарная метка
    anomaly_type: str           # "none" | "memory_leak" | ...
    severity: str               # "none" | "low" | "medium" | "high" | "critical"

    def as_dict(self) -> dict:
        """Конвертировать в плоский словарь для pandas / csv.DictWriter."""
        return {
            "timestamp": self.timestamp,
            "day_index": self.day_index,
            "weekday": self.weekday,
            "hour": round(self.hour, 3),
            "stage_name": self.stage_name,
            "mini_pattern": self.mini_pattern,
            "users": round(self.users, 1),
            "spawn_rate": round(self.spawn_rate, 2),
            "scale_factor": round(self.scale_factor, 4),
            "error_rate": round(self.error_rate, 5),
            "latency_ms": round(self.latency_ms, 1),
            "is_weekend": int(self.is_weekend),
            "is_anomaly": int(self.is_anomaly),
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
        }


def compute_latency(
    users: float,
    mini_pattern_name: Optional[str],
    anomaly: AnomalyEvent,
    base_latency_ms: float = 80.0,
) -> float:
    """Рассчитать latency_ms на основе нагрузки и аномалий.

    Используется ScenarioShape для заполнения MetricsTick.latency_ms.

    Формула:
        base → нагрузочная составляющая (квадратичная по users)
             + мультипликатор мини-паттерна
             + мультипликатор аномалии
             + гауссовый шум 5%

    Args:
        users: текущее число пользователей.
        mini_pattern_name: имя активного мини-паттерна.
        anomaly: текущий AnomalyEvent.
        base_latency_ms: базовая задержка при нулевой нагрузке.

    Returns:
        latency_ms > 0.
    """
    # Нагрузочная составляющая: квадратичный рост после 100 users
    load_factor = 1.0 + max(0.0, (users - 100) / 100) ** 1.6

    pattern_mult = 1.0
    if mini_pattern_name and mini_pattern_name in MINI_PATTERNS:
        pattern_mult = MINI_PATTERNS[mini_pattern_name].latency_multiplier

    noise = random.gauss(1.0, 0.05)
    latency = (
        base_latency_ms
        * load_factor
        * pattern_mult
        * anomaly.latency_multiplier
        * noise
    )
    return max(5.0, latency)


# ═══════════════════════════════════════════════════════════════════════
# Основные сценарии
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, list[dict]] = {

    # ------------------------------------------------------------------
    # dataset_day — 24-часовой цикл для формирования датасета
    # Стейджи идут последовательно, между ними плавный переход.
    # Внутри каждого стейджа вставляются мини-паттерны из пула.
    # ------------------------------------------------------------------
    "dataset_day": [
        # ── 1. Ночь ───────────────────────────────────────────────────
        {
            "name": "night",
            "duration_min": 4 * 3600,
            "duration_max": 6 * 3600,
            "users_min": 4, "users_max": 12,
            "spawn_min": 0.8, "spawn_max": 2.5,
            "transition": 300,
            "mini_patterns": [("night_watch", 10), ("lull", 8), ("batch_storm", 1)],
            "weights": {
                "NormalUser": 10,
                "MixedUser": 2,
                "AggressiveUser": 1,
                "InvalidUser": 0,
                "Chaos500User": 0,
            },
        },
        # ── 2. Ранний рассвет ─────────────────────────────────────────
        {
            "name": "early_dawn",
            "duration_min": 30 * 60,
            "duration_max": 60 * 60,
            "users_min": 6, "users_max": 18,
            "spawn_min": 1.0, "spawn_max": 3.5,
            "transition": 420,
            "mini_patterns": [("lull", 3), ("night_watch", 5)],
            "skip_probability": 0.30,
            "weights": {
                "NormalUser": 10,
                "MixedUser": 2,
                "AggressiveUser": 0,
                "InvalidUser": 0,
                "Chaos500User": 0,
            },
        },
        # ── 3. Утренний рост ──────────────────────────────────────────
        {
            "name": "morning_ramp",
            "duration_min": 90 * 60,
            "duration_max": 150 * 60,
            "users_min": 15, "users_max": 60,
            "spawn_min": 8.0, "spawn_max": 20.0,
            "transition": 480,
            "mini_patterns": [("wave", 4), ("lull", 2)],
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 2,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 4. Дневной пик ────────────────────────────────────────────
        {
            "name": "day_peak",
            "duration_min": 5 * 3600,
            "duration_max": 7 * 3600,
            "users_min": 75, "users_max": 115,
            "spawn_min": 18.0, "spawn_max": 38.0,
            "transition": 480,
            "mini_patterns": [("wave", 4), ("micro_spike", 2), ("lull", 1)],
            "weights": {
                "NormalUser": 6,
                "MixedUser": 4,
                "AggressiveUser": 4,
                "InvalidUser": 2,
                "Chaos500User": 1,
            },
        },
        # ── 5. Послеобеденный провал ──────────────────────────────────
        {
            "name": "lunch_dip",
            "duration_min": 45 * 60,
            "duration_max": 75 * 60,
            "users_min": 30, "users_max": 55,
            "spawn_min": 5.0, "spawn_max": 10.0,
            "transition": 360,
            "mini_patterns": [("lull", 4), ("wave", 2)],
            "skip_probability": 0.25,
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 1,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 6. Нагрузочный стресс ─────────────────────────────────────
        {
            "name": "stress",
            "duration_min": 2.5 * 3600,
            "duration_max": 3.5 * 3600,
            "users_min": 100, "users_max": 160,
            "spawn_min": 35.0, "spawn_max": 70.0,
            "transition": 360,
            "mini_patterns": [
                ("degraded", 3), ("pulse", 3), ("micro_spike", 2), ("recovery", 1)
            ],
            "weights": {
                "NormalUser": 4,
                "MixedUser": 4,
                "AggressiveUser": 6,
                "InvalidUser": 3,
                "Chaos500User": 3,
            },
        },
        # ── 7. Flash-sale ─────────────────────────────────────────────
        {
            "name": "flash_sale",
            "duration_min": 10 * 60,
            "duration_max": 20 * 60,
            "users_min": 140, "users_max": 220,
            "spawn_min": 60.0, "spawn_max": 100.0,
            "transition": 60,
            "mini_patterns": [("pulse", 4), ("micro_spike", 3), ("degraded", 2)],
            "skip_probability": 0.55,
            "weights": {
                "NormalUser": 7,
                "MixedUser": 4,
                "AggressiveUser": 3,
                "InvalidUser": 1,
                "Chaos500User": 1,
            },
        },
        # ── 8. Canary-рамп ────────────────────────────────────────────
        {
            "name": "canary_ramp",
            "duration_min": 20 * 60,
            "duration_max": 35 * 60,
            "users_min": 20, "users_max": 80,
            "spawn_min": 3.0, "spawn_max": 8.0,
            "transition": 600,
            "mini_patterns": [("wave", 1)],
            "skip_probability": 0.45,
            "weights": {
                "NormalUser": 9,
                "MixedUser": 3,
                "AggressiveUser": 1,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 9. Chaos window ───────────────────────────────────────────
        {
            "name": "chaos_window",
            "duration_min": 15 * 60,
            "duration_max": 25 * 60,
            "users_min": 160, "users_max": 200,
            "spawn_min": 28.0, "spawn_max": 55.0,
            "transition": 240,
            "mini_patterns": [
                ("degraded", 3), ("micro_spike", 3), ("recovery", 1), ("pulse", 2)
            ],
            "skip_probability": 0.20,
            "weights": {
                "NormalUser": 3,
                "MixedUser": 2,
                "AggressiveUser": 4,
                "InvalidUser": 3,
                "Chaos500User": 6,
            },
        },
        # ── 10. DDoS-имитация ─────────────────────────────────────────
        {
            "name": "ddos_sim",
            "duration_min": 5 * 60,
            "duration_max": 12 * 60,
            "users_min": 300, "users_max": 500,
            "spawn_min": 120.0, "spawn_max": 200.0,
            "transition": 30,
            "mini_patterns": [("degraded", 3), ("pulse", 4)],
            "skip_probability": 0.70,
            "weights": {
                "NormalUser": 1,
                "MixedUser": 1,
                "AggressiveUser": 6,
                "InvalidUser": 4,
                "Chaos500User": 8,
            },
        },
        # ── 11. Вечернее затухание ────────────────────────────────────
        {
            "name": "evening_wind_down",
            "duration_min": 90 * 60,
            "duration_max": 150 * 60,
            "users_min": 45, "users_max": 90,
            "spawn_min": 5.0, "spawn_max": 11.0,
            "transition": 600,
            "mini_patterns": [("lull", 3), ("wave", 2), ("night_watch", 4)],
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 2,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 12. Второй вечерний пик ───────────────────────────────────
        {
            "name": "evening_peak",
            "duration_min": 60 * 60,
            "duration_max": 90 * 60,
            "users_min": 55, "users_max": 95,
            "spawn_min": 10.0, "spawn_max": 22.0,
            "transition": 480,
            "mini_patterns": [("wave", 4), ("lull", 2), ("micro_spike", 1)],
            "skip_probability": 0.35,
            "weights": {
                "NormalUser": 7,
                "MixedUser": 4,
                "AggressiveUser": 3,
                "InvalidUser": 1,
                "Chaos500User": 1,
            },
        },
    ],

    # ------------------------------------------------------------------
    # quick_trial — ~20-минутный прогон для проверки работоспособности
    # ------------------------------------------------------------------
    "quick_trial": [
        {
            "name": "warmup",
            "duration_min": 90,
            "duration_max": 150,
            "users_min": 3, "users_max": 7,
            "spawn_min": 0.8, "spawn_max": 1.8,
            "transition": 45,
            "mini_patterns": [],
            "weights": {
                "NormalUser": 8,
                "MixedUser": 2,
                "AggressiveUser": 1,
                "InvalidUser": 0,
                "Chaos500User": 0,
            },
        },
        {
            "name": "load",
            "duration_min": 360,
            "duration_max": 480,
            "users_min": 15, "users_max": 28,
            "spawn_min": 3.0, "spawn_max": 5.5,
            "transition": 90,
            "mini_patterns": [("wave", 3), ("lull", 2)],
            "weights": {
                "NormalUser": 6,
                "MixedUser": 3,
                "AggressiveUser": 3,
                "InvalidUser": 1,
                "Chaos500User": 1,
            },
        },
        {
            "name": "spike",
            "duration_min": 90,
            "duration_max": 150,
            "users_min": 45, "users_max": 75,
            "spawn_min": 12.0, "spawn_max": 25.0,
            "transition": 60,
            "mini_patterns": [("pulse", 3), ("degraded", 2)],
            "weights": {
                "NormalUser": 4,
                "MixedUser": 3,
                "AggressiveUser": 5,
                "InvalidUser": 2,
                "Chaos500User": 3,
            },
        },
        {
            "name": "recovery",
            "duration_min": 150,
            "duration_max": 210,
            "users_min": 8, "users_max": 18,
            "spawn_min": 1.5, "spawn_max": 4.0,
            "transition": 90,
            "mini_patterns": [("recovery", 3), ("lull", 2)],
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 1,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        {
            "name": "chaos",
            "duration_min": 150,
            "duration_max": 210,
            "users_min": 20, "users_max": 38,
            "spawn_min": 5.0, "spawn_max": 9.0,
            "transition": 60,
            "mini_patterns": [("degraded", 3), ("micro_spike", 2)],
            "weights": {
                "NormalUser": 3,
                "MixedUser": 2,
                "AggressiveUser": 4,
                "InvalidUser": 3,
                "Chaos500User": 6,
            },
        },
    ],
}
