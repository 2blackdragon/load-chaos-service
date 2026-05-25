import requests
import csv
from datetime import datetime

# =========================
# Настройки
# =========================

PROMETHEUS_URL = "http://138.16.162.15:9090"

QUERY = """
sum by (handler, method, status) (
  rate(
    http_requests_total{
      handler!="none",
      handler!="/metrics"
    }[5m]
  )
)
"""

START = "2026-05-24T23:00:00Z"
END = "2026-05-25T18:00:00Z"

STEP = "15s"

OUTPUT_FILE = "metrics.csv"

# =========================
# Запрос к Prometheus
# =========================

url = f"{PROMETHEUS_URL}/api/v1/query_range"

params = {
    "query": QUERY,
    "start": START,
    "end": END,
    "step": STEP
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

data = response.json()

if data["status"] != "success":
    raise Exception(f"Prometheus API error: {data}")

results = data["data"]["result"]

# =========================
# Сохранение в CSV
# =========================

with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "timestamp",
        "datetime_utc",
        "handler",
        "method",
        "status",
        "rps"
    ])

    for series in results:
        metric = series["metric"]

        handler = metric.get("handler", "")
        method = metric.get("method", "")
        status = metric.get("status", "")

        for ts, value in series["values"]:
            ts_float = float(ts)

            writer.writerow([
                int(ts_float),
                datetime.utcfromtimestamp(ts_float).isoformat(),
                handler,
                method,
                status,
                round(float(value), 6)
            ])

print(f"Данные сохранены в {OUTPUT_FILE}")