import requests
import csv
from datetime import datetime
import os

# =========================
# Настройки
# =========================

PROMETHEUS_URL = "http://138.16.162.15:9090"

# Запрос суммирует по хендлерам все методы и статусы
QUERY = """
sum by (handler) (
  rate(
    http_requests_total{
      handler!="none",
      handler!="/metrics"
    }[$__rate_interval]
  )
)
"""

# Заменяем $__rate_interval на конкретное значение
RATE_INTERVAL = "5m"  # или "1m", "2m" - как в Grafana

FULL_QUERY = QUERY.replace("$__rate_interval", RATE_INTERVAL)

START = "2026-05-25T18:42:00Z"  # МСК 21:42
END = "2026-05-26T10:00:00Z"   # МСК 13:00

STEP = "15s"

url = f"{PROMETHEUS_URL}/api/v1/query_range"

params = {
    "query": FULL_QUERY,
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

if not results:
    print("Нет данных по запросу!")
    exit()


for series in results:
    handler = series["metric"].get("handler", "unknown")
    
    # Очищаем имя файла от недопустимых символов
    safe_handler_name = handler.replace("/", "_").replace("\\", "_")
    if safe_handler_name.startswith("_"):
        safe_handler_name = safe_handler_name[1:]
    
    filename = f"{safe_handler_name}_total_rps.csv"
    filepath = os.path.join("../data", filename)
    
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Заголовки
        writer.writerow([
            "timestamp",
            "datetime_utc",
            "datetime_msk",
            "handler",
            "total_rps"
        ])
        
        # Записываем все точки времени для этого handler'а
        for ts, value in series["values"]:
            ts_float = float(ts)
            
            # Конвертируем UTC в МСК (UTC+3)
            dt_utc = datetime.utcfromtimestamp(ts_float)
            dt_msk = datetime.fromtimestamp(ts_float + 3*3600)
            
            writer.writerow([
                int(ts_float),
                dt_utc.isoformat(),
                dt_msk.strftime("%Y-%m-%d %H:%M:%S"),
                handler,
                round(float(value), 6)
            ])
    
    # Получаем последнее значение RPS
    last_value = float(series["values"][-1][1]) if series["values"] else 0
    data_points = len(series["values"])
