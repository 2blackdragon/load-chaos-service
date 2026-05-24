from __future__ import annotations

import math
import os
import random
import time

from locust import events, LoadTestShape

from scenarios import MINI_PATTERNS, SCENARIOS, MiniPattern, build_stages
from users import AggressiveUser, Chaos500User, InvalidUser, MixedUser, NormalUser

# ---------------------------------------------------------------------------
# Конфиг из env
# ---------------------------------------------------------------------------

SCENARIO_NAME = os.getenv("LOCUST_SCENARIO", "quick_trial")

# build_stages() раскрывает сценарий: применяет skip_probability и фиксирует
# случайную duration для каждого стейджа на весь прогон.
SCENARIO = build_stages(SCENARIO_NAME)

# Шум на каждом тике ±N% от текущего таргета (поверх шума мини-паттерна)
GLOBAL_NOISE = float(os.getenv("LOCUST_NOISE", "0.06"))

ALL_USER_CLASSES = [NormalUser, AggressiveUser, MixedUser, InvalidUser, Chaos500User]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _smooth(t: float) -> float:
    """Smoothstep — плавнее чем линейная, без рывков на концах."""
    return t * t * (3 - 2 * t)


def _apply_noise(value: float, fraction: float) -> float:
    noise = random.uniform(-fraction, fraction)
    return max(1.0, value * (1 + noise))


def _set_weights(weights: dict[str, int]) -> None:
    mapping = {cls.__name__: cls for cls in ALL_USER_CLASSES}
    for name, cls in mapping.items():
        cls.weight = weights.get(name, 0)


def _resolve_duration(stage: dict) -> int:
    """Вернуть зафиксированную длительность стейджа.

    При первом вызове выбирает случайное значение из диапазона
    duration_min/duration_max и кеширует его в ключе _duration_resolved,
    чтобы длительность не менялась между тиками.
    Поддерживает старый формат с ключом duration (число).
    """
    if "_duration_resolved" not in stage:
        if "duration" in stage:
            stage["_duration_resolved"] = int(stage["duration"])
        else:
            lo = int(stage["duration_min"])
            hi = int(stage["duration_max"])
            stage["_duration_resolved"] = random.randint(lo, hi)
    return stage["_duration_resolved"]


# ---------------------------------------------------------------------------
# Состояние мини-паттернов
# ---------------------------------------------------------------------------


class MiniPatternState:
    """
    Выбирает мини-паттерны из пула стейджа взвешенным случайным выбором.

    Пул задаётся как список пар (имя, вес) или просто имён (тогда вес=1).
    Следующий паттерн выбирается через random.choices — чем выше вес,
    тем чаще паттерн встречается. Повторы подряд исключены: если выпал
    тот же паттерн что и текущий, делается ещё одна попытка (до 3 раз).

    Длительность паттерна фиксируется один раз при старте (_resolved_duration),
    чтобы повторные обращения к MiniPattern.duration не давали разные значения.
    """

    def __init__(self, pattern_spec: list):
        self._names: list[str] = []
        self._weights: list[float] = []
        self._current: MiniPattern | None = None
        self._started_at: float = 0.0
        self._resolved_duration: int = 0
        self.reset(pattern_spec)

    def reset(self, pattern_spec: list) -> None:
        """pattern_spec: list of str или list of (str, float)."""
        self._names = []
        self._weights = []
        for entry in pattern_spec:
            if isinstance(entry, (list, tuple)):
                name, w = entry[0], float(entry[1])
            else:
                name, w = entry, 1.0
            if name in MINI_PATTERNS:
                self._names.append(name)
                self._weights.append(w)
        self._current = None
        self._started_at = 0.0
        self._resolved_duration = 0

    def tick(self, now: float) -> MiniPattern | None:
        if not self._names:
            return None

        if self._current is None:
            self._start_next(now)

        elapsed = now - self._started_at
        if elapsed >= self._resolved_duration:
            self._start_next(now)

        return self._current

    def progress(self, now: float) -> float:
        if self._current is None or self._resolved_duration == 0:
            return 0.0
        elapsed = now - self._started_at
        return min(1.0, elapsed / self._resolved_duration)

    def _pick_name(self) -> str:
        """Взвешенный выбор без повтора текущего паттерна (до 3 попыток)."""
        current_name = self._current.name if self._current else None
        for _ in range(3):
            name = random.choices(self._names, weights=self._weights, k=1)[0]
            if name != current_name or len(self._names) == 1:
                return name
        return name  # после 3 попыток берём что есть

    def _start_next(self, now: float) -> None:
        name = self._pick_name()
        self._current = MINI_PATTERNS[name]
        self._started_at = now
        # Фиксируем длительность один раз — вызываем property ровно здесь
        self._resolved_duration = self._current.duration
        print(f"[MINI] → {self._current.name} ({self._resolved_duration}s, w={self._weights[self._names.index(name)]:.0f})")


# ---------------------------------------------------------------------------
# ScenarioShape
# ---------------------------------------------------------------------------


class ScenarioShape(LoadTestShape):
    """
    Управляет нагрузкой по стейджам из scenarios.py.

    Каждый тик:
    1. Определяет текущий стейдж и прогресс внутри него.
    2. Если идёт transition-период — плавная интерполяция через smoothstep.
    3. Если в стейдже есть mini_patterns — накладывает паттерн через sin-горб.
    4. Добавляет глобальный шум ±LOCUST_NOISE.
    5. Возвращает (users, spawn_rate).
    """

    def __init__(self):
        super().__init__()
        self._mini = MiniPatternState([])
        self._prev_stage_idx: int = -1
        self._weights_stage_idx: int = -1

    def tick(self) -> tuple[int, float] | None:
        run_time = self.get_run_time()
        now = time.time()

        # --- Найти текущий стейдж ---
        elapsed = 0.0
        stage_idx = None
        stage_start = 0.0

        for i, stage in enumerate(SCENARIO):
            duration = _resolve_duration(stage)
            stage_end = elapsed + duration
            if run_time < stage_end:
                stage_idx = i
                stage_start = elapsed
                break
            elapsed = stage_end

        if stage_idx is None:
            return None  # сценарий завершён

        stage = SCENARIO[stage_idx]
        stage_elapsed = run_time - stage_start
        transition = stage.get("transition", 60)

        # --- Обновить веса при смене стейджа ---
        if stage_idx != self._weights_stage_idx:
            _set_weights(stage.get("weights", {}))
            self._weights_stage_idx = stage_idx
            duration = _resolve_duration(stage)
            print(f"[STAGE] → {stage.get('name', stage_idx)}  duration={duration}s  weights={stage.get('weights')}")

        # --- Обновить пул мини-паттернов при смене стейджа ---
        if stage_idx != self._prev_stage_idx:
            self._mini.reset(stage.get("mini_patterns", []))
            self._prev_stage_idx = stage_idx

        # --- Базовый диапазон текущего стейджа ---
        u_min = stage["users_min"]
        u_max = stage["users_max"]
        s_min = stage["spawn_min"]
        s_max = stage["spawn_max"]

        # --- Transition: плавный вход из предыдущего стейджа ---
        if stage_elapsed < transition and stage_idx > 0:
            prev = SCENARIO[stage_idx - 1]
            t = _smooth(stage_elapsed / transition)
            prev_u = (prev["users_min"] + prev["users_max"]) / 2
            curr_u = (u_min + u_max) / 2
            prev_s = (prev["spawn_min"] + prev["spawn_max"]) / 2
            curr_s = (s_min + s_max) / 2
            base_users = _lerp(prev_u, curr_u, t)
            base_spawn = _lerp(prev_s, curr_s, t)
        else:
            base_users = random.uniform(u_min, u_max)
            base_spawn = random.uniform(s_min, s_max)

        # --- Мини-паттерн (sin-горб поверх базы) ---
        mini = self._mini.tick(now)
        if mini is not None:
            progress = self._mini.progress(now)
            sin_t = math.sin(math.pi * progress)  # 0 → 1 → 0

            mini_u = _lerp(mini.users_min, mini.users_max, sin_t)
            mini_s = _lerp(mini.spawn_min, mini.spawn_max, sin_t)

            # blend=0.4: паттерн влияет на 40% итогового значения
            blend = 0.4
            base_users = _lerp(base_users, mini_u, blend)
            base_spawn = _lerp(base_spawn, mini_s, blend)

            base_users = _apply_noise(base_users, mini.noise)
            base_spawn = _apply_noise(base_spawn, mini.noise)

        # --- Глобальный шум ---
        base_users = _apply_noise(base_users, GLOBAL_NOISE)
        base_spawn = _apply_noise(base_spawn, GLOBAL_NOISE * 0.5)

        users = max(1, int(round(base_users)))
        spawn_rate = max(0.5, round(base_spawn, 2))

        return users, spawn_rate


# ---------------------------------------------------------------------------
# Event hooks
# ---------------------------------------------------------------------------


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"[SCENARIO] Starting '{SCENARIO_NAME}' ({len(SCENARIO)} stages)")
    if SCENARIO:
        _set_weights(SCENARIO[0].get("weights", {}))
