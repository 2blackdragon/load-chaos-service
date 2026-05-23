SCENARIOS = {
    # "steady": [
    #     {"duration": 600, "users": 50, "spawn_rate": 5},
    # ],
    # "ramp_up": [
    #     {"duration": 30, "users": 5, "spawn_rate": 5},
    #     {"duration": 120, "users": 100, "spawn_rate": 20},
    #     {"duration": 120, "users": 250, "spawn_rate": 50},
    # ],
    # "spike": [
    #     {"duration": 15, "users": 10, "spawn_rate": 10},
    #     {"duration": 30, "users": 800, "spawn_rate": 400},
    #     {"duration": 60, "users": 40, "spawn_rate": 100},
    # ],
    # "stress": [
    #     {"duration": 60, "users": 50, "spawn_rate": 10},
    #     {"duration": 60, "users": 200, "spawn_rate": 40},
    #     {"duration": 120, "users": 500, "spawn_rate": 100},
    # ],
    # "soak": [
    #     {"duration": 3600, "users": 150, "spawn_rate": 25},
    # ],
    # "chaos": [
    #     {"duration": 10, "users": 5, "spawn_rate": 5},
    #     {"duration": 300, "users": 300, "spawn_rate": 150},
    # ],
    # # Composite continuous profile suitable for dataset collection
    # "continuous": [
    #     {"duration": 300, "users": 50, "spawn_rate": 10},  # warmup 5m
    #     {"duration": 1800, "users": 150, "spawn_rate": 30},  # steady 30m
    #     {"duration": 600, "users": 250, "spawn_rate": 50},  # upward wave
    #     {"duration": 600, "users": 100, "spawn_rate": 40},  # downward wave
    #     {"duration": 300, "users": 800, "spawn_rate": 400},  # short spike
    #     {"duration": 3600, "users": 200, "spawn_rate": 25},  # soak 1h
    # ],
    "dataset_day": [
        {
            "name": "night",
            "duration": 3 * 3600,
            "users": 20,
            "spawn_rate": 5,
            "weights": {
                "NormalUser": 10,
                "MixedUser": 2,
                "AggressiveUser": 1,
                "InvalidUser": 0,
                "Chaos500User": 0
            }
        },

        {
            "name": "morning",
            "duration": 2 * 3600,
            "users": 100,
            "spawn_rate": 20,
            "weights": {
                "NormalUser": 8,
                "MixedUser": 3,
                "AggressiveUser": 3,
                "InvalidUser": 1,
                "Chaos500User": 0
            }
        },

        {
            "name": "day_peak",
            "duration": 6 * 3600,
            "users": 250,
            "spawn_rate": 30,
            "weights": {
                "NormalUser": 6,
                "MixedUser": 4,
                "AggressiveUser": 5,
                "InvalidUser": 2,
                "Chaos500User": 1
            }
        },

        {
            "name": "stress",
            "duration": 3 * 3600,
            "users": 500,
            "spawn_rate": 60,
            "weights": {
                "NormalUser": 4,
                "MixedUser": 4,
                "AggressiveUser": 6,
                "InvalidUser": 3,
                "Chaos500User": 3
            }
        },

        {
            "name": "chaos_window",
            "duration": 2 * 3600,
            "users": 300,
            "spawn_rate": 50,
            "weights": {
                "NormalUser": 3,
                "MixedUser": 2,
                "AggressiveUser": 4,
                "InvalidUser": 3,
                "Chaos500User": 6
            }
        }
    ]
}
