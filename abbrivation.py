import pandas as pd

with open('output.csv', 'r') as f:
    data = pd.read_csv(f)
# Пусть data — твой датафрейм с колонкой 'Label'
normal_traffic = data[data['Label'] == 0]
attacks = data[data['Label'] != 0]

# Случайно берём 800000 нормального трафика (без замены)
normal_sampled = normal_traffic.sample(n=800000, random_state=42)

# Склеиваем обратно нормальный трафик (сокращённый) + все атаки
balanced_data = pd.concat([normal_sampled, attacks])

# Можно перемешать строки, чтобы порядок не влиял
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(balanced_data['Label'].value_counts())

balanced_data.to_csv('balanced_output.csv', index=False)
