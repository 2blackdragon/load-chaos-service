import pandas as pd

# Загружаем датасеты
df1 = pd.read_csv("../data/metrics_15s.csv")
df2 = pd.read_csv("../data/cpu_dataset.csv")

# Удаляем timestamp из первого датасета
if "timestamp" in df1.columns:
    df1 = df1.drop(columns=["timestamp"])

# Берем из первого датасета столько строк,
# сколько есть во втором
df1_slice = df1.iloc[15: 15 + len(df2)]

# Склеиваем ПО КОЛОНКАМ
merged_df = pd.concat(
    [
        df2.reset_index(drop=True),
        df1_slice.reset_index(drop=True)
    ],
    axis=1
)

# Сохраняем
merged_df.to_csv("merged_dataset.csv", index=False)

print("Размер первого среза:", df1_slice.shape)
print("Размер второго датасета:", df2.shape)
print("Размер итогового датасета:", merged_df.shape)
