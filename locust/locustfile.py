import os
from pathlib import Path
from locust import events, LoadTestShape
from scenarious import SCENARIOS

from users import (
    NormalUser,
    AggressiveUser,
    MixedUser,
    InvalidUser,
    Chaos500User,
)


SCENARIO_NAME = os.getenv("LOCUST_SCENARIO", "load")
SCENARIO = SCENARIOS.get(SCENARIO_NAME, SCENARIOS["load"])


class ScenarioShape(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()

        elapsed = 0
        for stage in SCENARIO:
            elapsed += stage["duration"]
            if run_time < elapsed:
                return (stage["users"], stage["spawn_rate"])

        return None



@events.test_start.add_listener
def apply_weights(environment, **kwargs):
    stage = SCENARIO[0] if "weights" not in SCENARIO[0] else SCENARIO[0]

    weights = stage.get("weights", None)
    if not weights:
        print("⚠️ No weights defined for scenario")
        return

    NormalUser.weight = weights.get("NormalUser", 5)
    AggressiveUser.weight = weights.get("AggressiveUser", 2)
    MixedUser.weight = weights.get("MixedUser", 3)
    InvalidUser.weight = weights.get("InvalidUser", 1)
    Chaos500User.weight = weights.get("Chaos500User", 1)

    print(f"[WEIGHTS] {weights}")
