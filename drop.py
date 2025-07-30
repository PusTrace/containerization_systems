import pandas as pd

# Чтение
data = pd.read_csv('output.csv')

# Список колонок для удаления
cols_to_drop = [
    # Колонки, которые ты уже выбрасывал
    "Flow ID",
    "Total TCP Flow Time",
    "Packet Length Variance",
    "Fwd Header Length",
    "Bwd Header Length",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
    "Src Port",
    "Dst Port",

    # Колонки с >95% нулей
    "Bwd RST Flags",
    "Fwd RST Flags",
    "Packet Length Min",
    "RST Flag Count",
    "Fwd Packet Length Min",
    "Bwd Packet Length Min",
    "Active Std",
    "Idle Std",
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Min",
    "Idle Mean",
    "Idle Max",
    # Дубли по bulk уже есть выше, так что их второй раз можно не писать
]



# Удаление только тех колонок, которые реально есть
existing_cols = [c for c in cols_to_drop if c in data.columns]
data.drop(columns=existing_cols, inplace=True)

# Разделение нормального трафика и атак
normal_traffic = data[data['Label'] == 0]
attacks = data[data['Label'] != 0]

# Ограничиваем нормальный трафик
normal_sampled = normal_traffic.sample(n=200000, random_state=42)

# Склеиваем обратно и перемешиваем
balanced_data = pd.concat([normal_sampled, attacks], axis=0)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Вывод статистики
print("Balanced dataset shape:", balanced_data.shape)
print("Class distribution:")
print(balanced_data['Label'].value_counts())

# Сохраняем результат
balanced_data.to_csv('balanced_output.csv', index=False)

print("Файл 'balanced_output.csv' успешно создан.")
