import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Загружаем данные из CSV
data = pd.read_csv(r'balanced_output.csv',
                   encoding='latin1', sep=';')

# Выделяем идентификатор потока и метку
target = data['Label']
y = data['Flow ID']

# Удаляем признаки, связанные со стендом
X = data.drop(['Flow ID', 'Label'], axis=1)  # Изменить набор данных

# Сохраняем имена оставшихся признаков
feature_names = X.columns

# Заменяем бесконечные значения на 0
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

# Удаляем признаки с небольшой дисперсией
selector = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)

# Получаем имена оставшихся признаков, корректируя индекс
reduced_feature_names = feature_names[selector.get_support()]

# Разделяем данные на обучающую и валидационную выборки (80% train, 20% valid)
X_train, X_temp, Y_train, Y_temp = train_test_split(X_reduced, target, test_size=0.2, random_state=10)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=10)  # 10% валид и 10% тест

# Выполняем SMOTE только на обучающей выборке
smote = SMOTE(sampling_strategy={3: 20000, 4: 20000, 5: 20000, 6: 20000,
                                   7: 20000, 8: 20000, 9: 20000, 10: 20000,
                                   11: 20000}, k_neighbors=5, random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

# Уменьшаем классы 0 и 1 до 100000
for label in [0, 2]:
    if (Y_resampled == label).sum() > 100000:
        indices = np.where(Y_resampled == label)[0]
        np.random.shuffle(indices)
        indices_to_delete = indices[100000:]
        X_resampled = np.delete(X_resampled, indices_to_delete, axis=0)
        Y_resampled = np.delete(Y_resampled, indices_to_delete)

# Объединяем обучающую выборку обратно с метками и идентификаторами
train_data = pd.DataFrame(X_resampled, columns=reduced_feature_names)
train_data['Label'] = pd.Series(Y_resampled).reset_index(drop=True)
train_data['Flow ID'] = y[y.index.isin(train_data.index)].reset_index(drop=True)  # Восстанавливаем Flow ID из изначальных данных

# Объединяем валидационную выборку
val_data = pd.DataFrame(X_val, columns=reduced_feature_names)
val_data['Label'] = Y_val.reset_index(drop=True)
val_data['Flow ID'] = y[y.index.isin(val_data.index)].reset_index(drop=True)  # Восстанавливаем Flow ID из изначальных данных

# Объединяем тестовую выборку
test_data = pd.DataFrame(X_test, columns=reduced_feature_names)
test_data['Label'] = Y_test.reset_index(drop=True)
test_data['Flow ID'] = y[y.index.isin(test_data.index)].reset_index(drop=True)  # Восстанавливаем Flow ID из изначальных данных

# Сохранение данных в файлы
train_data.to_csv('train.csv', index=False, sep=';')
val_data.to_csv('valid.csv', index=False, sep=';')
test_data.to_csv('test_data.csv', index=False, sep=';')

print("Files successfully saved")