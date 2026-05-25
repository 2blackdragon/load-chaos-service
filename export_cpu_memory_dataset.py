import requests
import csv

PROMETHEUS_URL = "http://138.16.162.15:9090"

START = "2026-05-24T23:00:00Z"
END = "2026-05-25T18:00:00Z"
STEP = "15s"

CPU_QUERY = """
100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])))
"""

MEM_QUERY = """
100 * (
  1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes
)
"""

def fetch(query):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    resp = requests.get(url, params={
        "query": query,
        "start": START,
        "end": END,
        "step": STEP
    })

    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success":
        raise Exception(data)

    return data["data"]["result"]

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

# CPU
cpu_results = fetch(CPU_QUERY)
save_csv("cpu_dataset.csv", cpu_results, "cpu_usage_percent")

# Memory
mem_results = fetch(MEM_QUERY)
save_csv("memory_dataset.csv", mem_results, "memory_usage_percent")

print("Done")
