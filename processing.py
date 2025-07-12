import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import kagglehub
import os

# Загрузка последней версии датасета
path = kagglehub.dataset_download("yigitsever/misuse-detection-in-containers-dataset")

print("Path to dataset files:", path)

# Загрузка данных
file_path = os.path.join(path, 'dataset.csv')
data = pd.read_csv(file_path, encoding='latin1', sep=',')

print(data.columns.tolist())


# Выделяем идентификатор потока и метку отдельно
flow_id = data['Flow ID']
labels = data['Label']

# Удаляем признаки, связанные со стендом
X = data.drop(['Flow ID', 'Src Port', 'Src IP', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Label'], axis=1)

# Заменяем бесконечные и NaN значения на 0
X = X.replace([np.inf, -np.inf], 0).fillna(0)

# Фильтрация признаков с низкой дисперсией
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

# Получаем имена оставшихся признаков
selected_features = X.columns[selector.get_support()]

# Финальный DataFrame
final_data = pd.DataFrame(X_reduced, columns=selected_features)

# Добавляем обратно Flow ID и Label
final_data['Flow ID'] = flow_id.reset_index(drop=True)
final_data['Label'] = labels.reset_index(drop=True)

# Сохраняем результат
final_data.to_csv('output.csv', index=False)

print("✅ File successfully saved with labels intact")
