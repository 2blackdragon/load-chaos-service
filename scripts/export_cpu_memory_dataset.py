import requests
import csv
from datetime import datetime, timedelta, timezone

PROMETHEUS_URL = "http://138.16.162.15:9090"

START = datetime.fromisoformat("2026-05-25T18:42:00+00:00")
END = datetime.fromisoformat("2026-05-28T06:00:00+00:00")

STEP = "15s"

CPU_QUERY = """
100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])))
"""

MEM_QUERY = """
100 * (
  1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes
)
"""

# размер одного чанка
CHUNK_HOURS = 6


def fetch_range(query, start_dt, end_dt):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    resp = requests.get(url, params={
        "query": query,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "step": STEP
    })

    resp.raise_for_status()

    data = resp.json()

    if data["status"] != "success":
        raise Exception(data)

    return data["data"]["result"]


def fetch_all(query):
    current = START

    merged = []

    while current < END:
        chunk_end = min(current + timedelta(hours=CHUNK_HOURS), END)

        print(f"Fetching {current} -> {chunk_end}")

        results = fetch_range(query, current, chunk_end)

        merged.extend(results)

        current = chunk_end

    return merged


def save_csv(filename, results, value_name):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            value_name
        ])

        for series in results:
            for ts, value in series["values"]:
                writer.writerow([
                    int(float(ts)),
                    float(value)
                ])


cpu_results = fetch_all(CPU_QUERY)
save_csv("../data/cpu_dataset.csv", cpu_results, "cpu_usage_percent")

mem_results = fetch_all(MEM_QUERY)
save_csv("../data/memory_dataset.csv", mem_results, "memory_usage_percent")

print("Done")