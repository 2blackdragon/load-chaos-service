from __future__ import annotations

import os
import random
import time
import csv
from pathlib import Path

from locust import events, LoadTestShape

from scenarios import (
    build_fourteen_day_plan,
    DayRun,
    WeeklyEnvelope,
    ANOMALY_INJECTOR,
    ERROR_MODEL,
    compute_latency,
    MetricsTick,
    MINI_PATTERNS,
)

from users import AggressiveUser, Chaos500User, InvalidUser, MixedUser, NormalUser

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENARIO_NAME = os.getenv("LOCUST_SCENARIO", "dataset_day")
START_WEEKDAY = int(os.getenv("LOCUST_START_WEEKDAY", "0"))
GLOBAL_NOISE = float(os.getenv("LOCUST_NOISE", "0.06"))
ALL_USER_CLASSES = [NormalUser, AggressiveUser, MixedUser, InvalidUser, Chaos500User]

# CSV output config
CSV_OUTPUT_DIR = Path(os.getenv("LOCUST_CSV_DIR", "."))
CSV_FILENAME = os.getenv("LOCUST_CSV_NAME", "locust_metrics.csv")

# Dataset tick interval (seconds)
DATASET_TICK_INTERVAL = float(os.getenv("LOCUST_DATASET_INTERVAL", "15.0"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def _apply_noise(value: float, fraction: float) -> float:
    noise = random.uniform(-fraction, fraction)
    return max(1.0, value * (1 + noise))


def _set_weights(weights: dict[str, int]) -> None:
    mapping = {cls.__name__: cls for cls in ALL_USER_CLASSES}
    for name, cls in mapping.items():
        cls.weight = weights.get(name, 0)


# ---------------------------------------------------------------------------
# CSV Writer
# ---------------------------------------------------------------------------

class MetricsCSVWriter:
    """Потоковая запись метрик в CSV без накопления в памяти."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file = None
        self._writer = None
        self._header_written = False

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(MetricsTick.__dataclass_fields__.keys()),
            )

    def write(self, tick: MetricsTick):
        self._ensure_open()
        if not self._header_written:
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(tick.as_dict())
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# 14-day auto runner
# ---------------------------------------------------------------------------

class ScenarioShape(LoadTestShape):
    """Auto 14-day execution inside Locust (no bash loop needed)."""

    def __init__(self):
        super().__init__()

        self.plan: list[DayRun] = build_fourteen_day_plan(
            start_weekday=START_WEEKDAY,
            scenario_name=SCENARIO_NAME,
        )

        # precompute day offsets
        self._day_offsets = []
        total = 0
        for day in self.plan:
            day_duration = 0
            for s in day.stages:
                day_duration += (s.get("_duration_resolved") or 0)
            self._day_offsets.append(total)
            total += day_duration

        self._total_duration = total
        self._weights_stage_idx = -1

        # CSV writer
        csv_path = CSV_OUTPUT_DIR / CSV_FILENAME
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_writer = MetricsCSVWriter(csv_path)
        print(f"[CSV] Writing metrics to: {csv_path.absolute()}")
        print(f"[CSV] Dataset tick interval: {DATASET_TICK_INTERVAL}s")

        # track mini-pattern state
        self._current_mini_pattern: str = ""
        self._mini_pattern_end_time: float = 0.0

        # ⬇️ Таймер для записи в CSV каждые N секунд
        self._last_dataset_tick: float = 0.0

    def _pick_mini_pattern(self, stage: dict, stage_start: float) -> str:
        """Выбрать мини-паттерн из стейджа по весам."""
        patterns = stage.get("mini_patterns", [])
        if not patterns:
            return ""

        names = [p[0] for p in patterns]
        weights = [p[1] for p in patterns]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for name, w in zip(names, weights):
            cumulative += w
            if r <= cumulative:
                return name
        return names[-1]

    def tick(self):
        run_time = self.get_run_time()

        # stop after 14 days
        if run_time > self._total_duration:
            self.csv_writer.close()
            return None

        # find day
        day_idx = 0
        for i in range(len(self._day_offsets)):
            if i == len(self._day_offsets) - 1 or run_time < self._day_offsets[i + 1]:
                day_idx = i
                break

        day = self.plan[day_idx]
        envelope = day.envelope

        # local time inside day
        day_start = self._day_offsets[day_idx]
        t_in_day = run_time - day_start

        # find stage
        elapsed = 0.0
        stage_idx = 0
        stage_start = 0.0

        for i, stage in enumerate(day.stages):
            duration = stage.get("_duration_resolved", 0)
            if t_in_day < elapsed + duration:
                stage_idx = i
                stage_start = elapsed
                break
            elapsed += duration

        stage = day.stages[stage_idx]
        stage_elapsed = t_in_day - stage_start
        transition = stage.get("transition", 60)

        # weights
        if stage_idx != self._weights_stage_idx:
            _set_weights(stage.get("weights", {}))
            self._weights_stage_idx = stage_idx
            print(f"[DAY {day.day_index}] STAGE {stage.get('name')}")

        u_min, u_max = stage["users_min"], stage["users_max"]
        s_min, s_max = stage["spawn_min"], stage["spawn_max"]

        # transition smoothing
        if stage_elapsed < transition and stage_idx > 0:
            prev = day.stages[stage_idx - 1]
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

        # envelope (14-day)
        scale = envelope.scale_factor
        base_users *= scale
        base_spawn *= scale

        # anomaly
        anomaly = ANOMALY_INJECTOR.tick(base_users, stage)
        base_users *= anomaly.users_multiplier
        base_spawn *= anomaly.latency_multiplier

        # noise
        base_users = _apply_noise(base_users, GLOBAL_NOISE)
        base_spawn = _apply_noise(base_spawn, GLOBAL_NOISE * 0.5)

        users = max(1, int(round(base_users)))
        spawn_rate = max(0.5, round(base_spawn, 2))

        # mini-pattern logic
        stage_abs_start = day_start + stage_start
        if run_time >= self._mini_pattern_end_time:
            self._current_mini_pattern = self._pick_mini_pattern(stage, stage_abs_start)
            if self._current_mini_pattern in MINI_PATTERNS:
                mp = MINI_PATTERNS[self._current_mini_pattern]
                self._mini_pattern_end_time = run_time + mp.duration
            else:
                self._mini_pattern_end_time = run_time + 60

        # compute error_rate and latency
        error_rate = ERROR_MODEL.compute(
            users=users,
            mini_pattern_name=self._current_mini_pattern or None,
            anomaly_multiplier=anomaly.error_multiplier,
        )
        latency_ms = compute_latency(
            users=users,
            mini_pattern_name=self._current_mini_pattern or None,
            anomaly=anomaly,
        )

        # ⬇️ ЗАПИСЬ В CSV ТОЛЬКО КАЖДЫЕ 15 СЕКУНД
        should_write_csv = (run_time - self._last_dataset_tick) >= DATASET_TICK_INTERVAL

        if should_write_csv:
            self._last_dataset_tick = run_time

            tick_data = MetricsTick(
                timestamp=time.time(),
                day_index=day.day_index,
                weekday=day.weekday,
                hour=(t_in_day % 86400) / 3600,
                stage_name=stage.get("name", ""),
                mini_pattern=self._current_mini_pattern,
                users=users,
                spawn_rate=spawn_rate,
                scale_factor=envelope.scale_factor,
                error_rate=error_rate,
                latency_ms=latency_ms,
                is_weekend=envelope.is_weekend,
                is_anomaly=anomaly.is_anomaly,
                anomaly_type=anomaly.label,
                severity=anomaly.severity.value,
            )
            self.csv_writer.write(tick_data)

            print(
                f"[DATASET] day={day.day_index} stage={stage.get('name')} "
                f"users={users} spawn={spawn_rate} err={error_rate:.4f} "
                f"lat={latency_ms:.1f}ms anomaly={anomaly.label}"
            )

        # Возвращаем управление Locust каждый тик (нагрузка работает непрерывно)
        print(f"[TICK] users={users}, spawn_rate={spawn_rate}")
        return users, spawn_rate


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("[SCENARIO] 14-day auto-run enabled")
    print(f"[SCENARIO] CSV output: {CSV_OUTPUT_DIR / CSV_FILENAME}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("[SCENARIO] Test finished, CSV closed")
