import os
import json
import time
import threading
import random

from locust import HttpUser, between, LoadTestShape, events

from users import (
    NormalUser,
    AggressiveUser,
    MixedUser,
    InvalidUser,
)


# Load local env file if present (simple KEY=VALUE parser)
def _load_local_env(path: str = "locust.env"):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # Do not overwrite existing environment vars
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


# attempt to load local env file in the locust dir
_load_local_env(os.path.join(os.path.dirname(__file__), "locust.env"))


# Define reusable scenarios. Pick one by setting the env var `LOCUST_SCENARIO`.
# Example: `LOCUST_SCENARIO=spike locust -f locustfile.py --host=http://localhost:8000`
SCENARIOS = {
    "steady": [
        {"duration": 600, "users": 50, "spawn_rate": 5},
    ],
    "ramp_up": [
        {"duration": 30, "users": 5, "spawn_rate": 5},
        {"duration": 120, "users": 100, "spawn_rate": 20},
        {"duration": 120, "users": 250, "spawn_rate": 50},
    ],
    "spike": [
        {"duration": 15, "users": 10, "spawn_rate": 10},
        {"duration": 30, "users": 800, "spawn_rate": 400},
        {"duration": 60, "users": 40, "spawn_rate": 100},
    ],
    "stress": [
        {"duration": 60, "users": 50, "spawn_rate": 10},
        {"duration": 60, "users": 200, "spawn_rate": 40},
        {"duration": 120, "users": 500, "spawn_rate": 100},
    ],
    "soak": [
        {"duration": 3600, "users": 150, "spawn_rate": 25},
    ],
    "chaos": [
        {"duration": 10, "users": 5, "spawn_rate": 5},
        {"duration": 300, "users": 300, "spawn_rate": 150},
    ],
    # Composite continuous profile suitable for dataset collection
    "continuous": [
        {"duration": 300, "users": 50, "spawn_rate": 10},  # warmup 5m
        {"duration": 1800, "users": 150, "spawn_rate": 30},  # steady 30m
        {"duration": 600, "users": 250, "spawn_rate": 50},  # upward wave
        {"duration": 600, "users": 100, "spawn_rate": 40},  # downward wave
        {"duration": 300, "users": 800, "spawn_rate": 400},  # short spike
        {"duration": 3600, "users": 200, "spawn_rate": 25},  # soak 1h
    ],
}


DEFAULT_SCENARIO_NAME = os.environ.get("LOCUST_SCENARIO", "ramp_up")
SCENARIO = SCENARIOS.get(DEFAULT_SCENARIO_NAME, SCENARIOS["ramp_up"])


class ScenarioShape(LoadTestShape):
    """Generic stage-based load shape.

    The shape reads stages from `SCENARIO` (a list of dicts with `duration`,
    `users`, `spawn_rate`) and returns the appropriate (user_count, spawn_rate)
    for the given elapsed time.
    """

    def tick(self):
        run_time = self.get_run_time()

        elapsed = run_time
        total = 0
        for stage in SCENARIO:
            total += stage["duration"]
            if elapsed < total:
                return (stage["users"], stage["spawn_rate"])

        return None


# --- Dataset logging ---
DATASET_PATH = os.environ.get("LOCUST_DATASET_PATH", "locust_dataset.jsonl")
LOG_SAMPLE_RATE = float(os.environ.get("LOCUST_LOG_SAMPLE_RATE", "1.0"))
METRICS_PATH = os.environ.get("LOCUST_METRICS_PATH", "locust_metrics.jsonl")
METRICS_INTERVAL = int(os.environ.get("LOCUST_METRICS_INTERVAL", "30"))


_dataset_lock = threading.Lock()
_dataset_file = None

def _open_dataset_file():
    global _dataset_file
    if _dataset_file is None:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True) if os.path.dirname(DATASET_PATH) else None
        _dataset_file = open(DATASET_PATH, "a", encoding="utf-8")
    return _dataset_file

def _log_record(rec: dict):
    if random.random() > LOG_SAMPLE_RATE:
        return

    f = _open_dataset_file()
    with _dataset_lock:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


# --- Metrics aggregator (per-minute) ---
_metrics_lock = threading.Lock()
_metrics_buffer: dict = {}
_metrics_file = None

def _open_metrics_file():
    global _metrics_file
    if _metrics_file is None:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True) if os.path.dirname(METRICS_PATH) else None
        _metrics_file = open(METRICS_PATH, "a", encoding="utf-8")
    return _metrics_file

def _record_metric(name: str, response_time: float, failed: bool):
    """Record a single request metric into current interval bucket."""
    bucket = int(time.time() // METRICS_INTERVAL) * METRICS_INTERVAL
    with _metrics_lock:
        m = _metrics_buffer.setdefault(bucket, {})
        entry = m.setdefault(name, {"rts": [], "count": 0, "failures": 0})
        entry["rts"].append(response_time)
        entry["count"] += 1
        if failed:
            entry["failures"] += 1

def _compute_percentiles(values, ps=(50,95,99)):
    if not values:
        return {f"p{p}": None for p in ps}
    vals = sorted(values)
    n = len(vals)
    out = {}
    for p in ps:
        k = int(round(p/100.0 * (n-1)))
        out[f"p{p}"] = vals[k]
    return out

def _flush_metrics_loop(interval: int = METRICS_INTERVAL):
    """Background thread: every `interval` seconds flush completed buckets to file."""
    while True:
        time.sleep(max(0.1, interval))

        now = time.time()
        current_bucket = int(now // interval) * interval
        cutoff = current_bucket - interval  # flush buckets strictly older than current
        to_flush = []
        with _metrics_lock:
            for bucket in list(_metrics_buffer.keys()):
                if bucket <= cutoff:
                    to_flush.append((bucket, _metrics_buffer.pop(bucket)))

        if not to_flush:
            continue

        f = _open_metrics_file()
        for bucket, payload in sorted(to_flush):
            # aggregate overall and per-name
            overall = {"bucket_start": bucket, "interval_s": interval, "total_requests": 0, "total_failures": 0, "per_name": {}}
            for name, data in payload.items():
                rts = data.get("rts", [])
                count = data.get("count", 0)
                failures = data.get("failures", 0)
                p = _compute_percentiles(rts)
                avg = sum(rts)/len(rts) if rts else None
                overall["per_name"][name] = {"count": count, "failures": failures, "avg_ms": avg, **p}
                overall["total_requests"] += count
                overall["total_failures"] += failures

            overall["rps"] = overall["total_requests"] / float(interval)
            with threading.Lock():
                f.write(json.dumps(overall, ensure_ascii=False) + "\n")
                f.flush()


# start background flusher thread
_metrics_thread = threading.Thread(target=_flush_metrics_loop, args=(METRICS_INTERVAL,), daemon=True)
_metrics_thread.start()


@events.request.add_listener
def on_request(request_type=None, name=None, response_time=None, response_length=None, response=None, context=None, exception=None, **kwargs):
    # support different locust versions by using a single request event
    user = kwargs.get("user") or (context and context.get("user"))
    user_type = type(user).__name__ if user is not None else None

    failed = bool(exception)
    if failed:
        rec = {
            "ts": time.time(),
            "request_type": request_type,
            "name": name,
            "response_time_ms": response_time,
            "status": "FAIL",
            "error": str(exception),
            "user_type": user_type,
        }
        # always log failures regardless of sample rate
        _log_record(rec)
    else:
        rec = {
            "ts": time.time(),
            "request_type": request_type,
            "name": name,
            "response_time_ms": response_time,
            "response_length": response_length,
            "status": "OK",
            "user_type": user_type,
        }
        _log_record(rec)

    # record metrics into aggregator (response_time in ms)
    try:
        rt_ms = float(response_time) if response_time is not None else None
        if rt_ms is not None and name is not None:
            _record_metric(name, rt_ms, failed)
    except Exception:
        pass
