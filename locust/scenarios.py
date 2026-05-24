"""
scenarios.py — пул сценариев и мини-паттернов для ScenarioShape

Каждый стейдж поддерживает:
    users_min / users_max       — диапазон пользователей (тик берёт случайное значение)
    spawn_min / spawn_max       — диапазон spawn_rate
    duration_min / duration_max — диапазон длительности стейджа в секундах;
                                  конкретное значение выбирается при старте прогона
    transition                  — секунды плавного перехода из предыдущего стейджа
    weights                     — веса классов пользователей
    mini_patterns               — список пар (имя, вес) или просто имён мини-паттернов.
                                  Формат: [("wave", 3), ("lull", 1)] или ["wave", "lull"].
                                  При простом списке все паттерны равновероятны.
    skip_probability            — вероятность пропуска стейджа при каждом прогоне
                                  (float 0.0–1.0; 0.0 = никогда, 1.0 = всегда;
                                  отсутствие ключа = 0.0, стейдж обязательный).

Для раскрытия сценария с учётом пропусков используйте build_stages():
    stages = build_stages("dataset_day")   # list[dict] без пропущенных стейджей

Мини-паттерны (MINI_PATTERNS) — короткие вставки (30–300 сек) с конкретным
характером нагрузки. ScenarioShape вытаскивает их из пула циклически внутри
каждого стейджа, у которого задан ключ mini_patterns.
Длительность каждого мини-паттерна тоже выбирается случайно из диапазона
duration_min / duration_max.
"""

from __future__ import annotations
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# build_stages — раскрытие сценария с вероятностным пропуском стейджей
# ---------------------------------------------------------------------------

def build_stages(scenario_name: str) -> list[dict]:
    """Вернуть список стейджей для прогона, учитывая skip_probability.

    Для каждого стейджа с ключом ``skip_probability`` бросается монета:
    если случайное число < skip_probability — стейдж пропускается.
    Стейджи без этого ключа (обязательные) включаются всегда.

    При пропуске стейджа transition следующего за ним стейджа автоматически
    увеличивается вдвое, чтобы переход оставался плавным.

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
        if prev_was_skipped and "transition" in s:
            s["transition"] = s["transition"] * 2
            log.debug(
                "stage '%s': transition doubled to %ds (prev stage was skipped)",
                s["name"], s["transition"],
            )
        result.append(s)
        prev_was_skipped = False

    return result


# ---------------------------------------------------------------------------
# Мини-паттерны
# ---------------------------------------------------------------------------

@dataclass
class MiniPattern:
    name: str
    duration_min: int       # минимальная длительность, секунд
    duration_max: int       # максимальная длительность, секунд
    users_min: int
    users_max: int
    spawn_min: float
    spawn_max: float
    noise: float = 0.04     # ±% шума на каждом тике (меньше = плавнее)
    weight: float = 1.0     # базовый вес для взвешенного выбора (переопределяется
                            # стейджем через mini_patterns)

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
    ),
    # Микро-спайк — короткий резкий всплеск
    "micro_spike": MiniPattern(
        name="micro_spike",
        duration_min=45, duration_max=90,
        users_min=70, users_max=130,
        spawn_min=25.0, spawn_max=50.0,
        noise=0.08,
    ),
    # Волна — плавный рост и спад
    "wave": MiniPattern(
        name="wave",
        duration_min=150, duration_max=240,
        users_min=30, users_max=100,
        spawn_min=5.0, spawn_max=12.0,
        noise=0.03,
    ),
    # Пульс — несколько быстрых всплесков подряд
    "pulse": MiniPattern(
        name="pulse",
        duration_min=75, duration_max=120,
        users_min=150, users_max=190,
        spawn_min=18.0, spawn_max=45.0,
        noise=0.10,
    ),
    # Агония — высокая нагрузка с умеренным шумом (degraded period)
    "degraded": MiniPattern(
        name="degraded",
        duration_min=120, duration_max=180,
        users_min=180, users_max=350,
        spawn_min=35.0, spawn_max=70.0,
        noise=0.12,
    ),
    # Восстановление — плавный спад после spike
    "recovery": MiniPattern(
        name="recovery",
        duration_min=100, duration_max=150,
        users_min=10, users_max=35,
        spawn_min=2.0, spawn_max=4.5,
        noise=0.03,
    ),
    # Ночное дежурство — совсем тихо, один-два пользователя
    "night_watch": MiniPattern(
        name="night_watch",
        duration_min=240, duration_max=360,
        users_min=2, users_max=7,
        spawn_min=0.3, spawn_max=0.9,
        noise=0.02,
    ),
    # Batch-шторм — ровная высокая нагрузка без шума (ночные расчёты)
    "batch_storm": MiniPattern(
        name="batch_storm",
        duration_min=200, duration_max=280,
        users_min=55, users_max=75,
        spawn_min=3.0, spawn_max=5.0,
        noise=0.02,
    ),
}


# ---------------------------------------------------------------------------
# Основные сценарии
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, list[dict]] = {

    # ------------------------------------------------------------------
    # dataset_day — 24-часовой цикл для формирования датасета
    # Стейджи идут последовательно, между ними плавный переход.
    # Внутри каждого стейджа вставляются мини-паттерны из пула.
    # ------------------------------------------------------------------
    "dataset_day": [
        # ── 1. Ночь ────────────────────────────────────────────────────────────
        {
            "name": "night",
            "duration_min": 4 * 3600,
            "duration_max": 6 * 3600,
            "users_min": 4, "users_max": 12,
            "spawn_min": 0.8, "spawn_max": 2.5,
            "transition": 300,
            "mini_patterns": [("night_watch", 4), ("lull", 3), ("batch_storm", 2)],
            "weights": {
                "NormalUser": 10,
                "MixedUser": 2,
                "AggressiveUser": 1,
                "InvalidUser": 0,
                "Chaos500User": 0,
            },
        },
        # ── 2. Ранний рассвет — первые одиночки, очень тихо ───────────────────
        # Вставляется перед morning_ramp: трафик чуть выше ночи, но без скачков.
        {
            "name": "early_dawn",
            "duration_min": 30 * 60,
            "duration_max": 60 * 60,
            "users_min": 6, "users_max": 18,
            "spawn_min": 1.0, "spawn_max": 3.5,
            "transition": 420,          # очень плавный переход от ночи
            "mini_patterns": [("lull", 3), ("night_watch", 5)],
            "skip_probability": 0.30,  # иногда ночь сразу переходит в утренний рост
            "weights": {
                "NormalUser": 10,
                "MixedUser": 2,
                "AggressiveUser": 0,
                "InvalidUser": 0,
                "Chaos500User": 0,
            },
        },
        # ── 3. Утренний рост ────────────────────────────────────────────────────
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
        # ── 4. Дневной пик ──────────────────────────────────────────────────────
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
        # ── 5. Послеобеденный провал — народ ушёл на обед ──────────────────────
        # Вставляется после day_peak: нагрузка падает на 40–50 %, затем плавно
        # возвращается обратно к stress-стейджу.
        {
            "name": "lunch_dip",
            "duration_min": 45 * 60,
            "duration_max": 75 * 60,
            "users_min": 30, "users_max": 55,
            "spawn_min": 5.0, "spawn_max": 10.0,
            "transition": 360,
            "mini_patterns": [("lull", 4), ("wave", 2)],
            "skip_probability": 0.25,  # провал не всегда выражен
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 1,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 6. Нагрузочный стресс ───────────────────────────────────────────────
        {
            "name": "stress",
            "duration_min": 2.5 * 3600,
            "duration_max": 3.5 * 3600,
            "users_min": 100, "users_max": 160,
            "spawn_min": 35.0, "spawn_max": 70.0,
            "transition": 360,
            "mini_patterns": [("degraded", 3), ("pulse", 3), ("micro_spike", 2), ("recovery", 1)],
            "weights": {
                "NormalUser": 4,
                "MixedUser": 4,
                "AggressiveUser": 6,
                "InvalidUser": 3,
                "Chaos500User": 3,
            },
        },
        # ── 7. Flash-sale — внезапный краткий всплеск (акция, рассылка) ─────────
        # Короткий стейдж с резким ростом spawn_rate и большим разбросом users.
        # Переход намеренно короткий — имитирует мгновенный приток трафика.
        {
            "name": "flash_sale",
            "duration_min": 10 * 60,
            "duration_max": 20 * 60,
            "users_min": 140, "users_max": 220,
            "spawn_min": 60.0, "spawn_max": 100.0,
            "transition": 60,           # резкий вход
            "mini_patterns": [("pulse", 4), ("micro_spike", 3), ("degraded", 2)],
            "skip_probability": 0.55,  # акция — редкое событие
            "weights": {
                "NormalUser": 7,
                "MixedUser": 4,
                "AggressiveUser": 3,
                "InvalidUser": 1,
                "Chaos500User": 1,
            },
        },
        # ── 8. Canary — постепенный рост с плановым откатом ────────────────────
        # Имитирует canary-деплой: нагрузка медленно растёт, потом так же
        # медленно снижается (transition на следующем стейдже сделает это).
        # noise низкий — нужна стабильная кривая для анализа.
        {
            "name": "canary_ramp",
            "duration_min": 20 * 60,
            "duration_max": 35 * 60,
            "users_min": 20, "users_max": 80,
            "spawn_min": 3.0, "spawn_max": 8.0,
            "transition": 600,          # очень плавный вход — 10 мин
            "mini_patterns": [("wave", 1)],
            "skip_probability": 0.45,  # только при плановых деплоях
            "weights": {
                "NormalUser": 9,
                "MixedUser": 3,
                "AggressiveUser": 1,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 9. Chaos window — хаос перед финалом ───────────────────────────────
        {
            "name": "chaos_window",
            "duration_min": 15 * 60,
            "duration_max": 25 * 60,
            "users_min": 160, "users_max": 200,
            "spawn_min": 28.0, "spawn_max": 55.0,
            "transition": 240,
            "mini_patterns": [("degraded", 3), ("micro_spike", 3), ("recovery", 1), ("pulse", 2)],
            "skip_probability": 0.20,  # хаос бывает не каждый день
            "weights": {
                "NormalUser": 3,
                "MixedUser": 2,
                "AggressiveUser": 4,
                "InvalidUser": 3,
                "Chaos500User": 6,
            },
        },
        # ── 10. DDoS-имитация — кратковременная сверхнагрузка ──────────────────
        # Очень высокий spawn_rate, почти все пользователи агрессивные/хаотичные.
        # Переход резкий на входе, длинный на выходе — система должна устоять
        # и плавно вернуться к норме.
        {
            "name": "ddos_sim",
            "duration_min": 5 * 60,
            "duration_max": 12 * 60,
            "users_min": 300, "users_max": 500,
            "spawn_min": 120.0, "spawn_max": 200.0,
            "transition": 30,           # почти мгновенный вход
            "mini_patterns": [("degraded", 3), ("pulse", 4)],
            "skip_probability": 0.70,  # DDoS — очень редкое событие
            "weights": {
                "NormalUser": 1,
                "MixedUser": 1,
                "AggressiveUser": 6,
                "InvalidUser": 4,
                "Chaos500User": 8,
            },
        },
        # ── 11. Вечернее затухание ──────────────────────────────────────────────
        {
            "name": "evening_wind_down",
            "duration_min": 90 * 60,
            "duration_max": 150 * 60,
            "users_min": 45, "users_max": 90,
            "spawn_min": 5.0, "spawn_max": 11.0,
            "transition": 600,          # длинный мягкий переход после DDoS
            "mini_patterns": [("lull", 3), ("wave", 2), ("night_watch", 4)],
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 2,
                "InvalidUser": 1,
                "Chaos500User": 0,
            },
        },
        # ── 12. Второй вечерний пик — возврат после ужина ──────────────────────
        # Характерен для развлекательных/медиа-сервисов: пользователи
        # возвращаются вечером, нагрузка чуть ниже дневного пика, но стабильнее.
        {
            "name": "evening_peak",
            "duration_min": 60 * 60,
            "duration_max": 90 * 60,
            "users_min": 55, "users_max": 95,
            "spawn_min": 10.0, "spawn_max": 22.0,
            "transition": 480,
            "mini_patterns": [("wave", 4), ("lull", 2), ("micro_spike", 1)],
            "skip_probability": 0.35,  # вечерний пик характерен не для всех сервисов
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
            "mini_patterns": [("wave", 3), ("lull", 2)],  # micro_spike убрали для плавности
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
