import pandas as pd
import os
import glob


# Получаем список всех CSV файлов
all_files = glob.glob(os.path.join('data/', '*.csv'))
all_files = [f for f in all_files if 'metrics_15s' not in os.path.basename(f)]

print(f"Найдено файлов: {len(all_files)}")
for f in all_files:
    print(f"  - {os.path.basename(f)}")

# ===== 2. ЗАГРУЖАЕМ И СКЛЕИВАЕМ ВСЕ ФАЙЛЫ =====
merged_df = None

for file_path in all_files:
    df = pd.read_csv(file_path)
    
    # Объединяем
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')



if merged_df is not None:
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    merged_df.to_csv('merged_all_files.csv', index=False)
    
    print(f"\nРезультат: {merged_df.shape[0]} строк, {merged_df.shape[1]} колонок")
    print(f"Колонки: {list(merged_df.columns)}")
    print(f"\nПервые 5 строк:")
    print(merged_df.head())
else:
    print("Файлы не найдены!")
