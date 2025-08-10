from sklearn.model_selection import train_test_split
import pandas as pd

# Загружаем данные
balanced_data = pd.read_csv('balanced_output.csv')

# Убираем нормальный трафик
balanced_data = balanced_data[balanced_data['Label'] != 0].reset_index(drop=True)

# удаляем минорные классы
balanced_data = balanced_data[~balanced_data['Label'].isin([10, 5, 9, 4, 7])]

# Разделение X/y
X = balanced_data.drop(columns=['Label'])
y = balanced_data['Label']

# Сплиты
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

# Сохраняем валидацию и тест
X_val['Label'] = y_val
X_test['Label'] = y_test
X_val.to_csv('gan_smote/validation.csv', index=False)
X_test.to_csv('gan_smote/test.csv', index=False)

# Создаем train_df
train_df = X_train.copy()
train_df['Label'] = y_train

magority_classes = [1, 2, 11]

for cls in magority_classes:
    # Разделение нормального трафика и атак
    target_traffic = train_df[train_df['Label'] == cls]
    other = train_df[train_df['Label'] != cls]

    # Ограничиваем нормальный трафик
    normal_sampled = target_traffic.sample(n=2000, random_state=42)

    # Склеиваем обратно и перемешиваем
    train_df = pd.concat([normal_sampled, other], axis=0)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем итог
train_df.to_csv('gan_smote/gan_augmented_train.csv', index=False)
