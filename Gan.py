from sklearn.model_selection import train_test_split
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata, Metadata  # <-- обновленный класс

# Загружаем данные
data = pd.read_csv('output.csv')

balanced_data = data.drop(columns=["Flow ID"])


# Убираем нормальный трафик

magority_classes = [0, 1, 2, 11]
for cls in magority_classes:
    # Разделение нормального трафика и атак
    target_traffic = balanced_data[balanced_data['Label'] == cls]
    other = balanced_data[balanced_data['Label'] != cls]

    # Ограничиваем нормальный трафик
    normal_sampled = target_traffic.sample(n=2000, random_state=42)

    # Склеиваем обратно и перемешиваем
    balanced_data = pd.concat([normal_sampled, other], axis=0)
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# удаляем минорные классы
# balanced_data = balanced_data[~balanced_data['Label'].isin([10,5,9])]

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
X_val.to_csv('datasets/validation.csv', index=False)
X_test.to_csv('datasets/test.csv', index=False)

# Создаем train_df
train_df = X_train.copy()
train_df['Label'] = y_train

# Классы для GAN (исключаем очень редкие и мажоритные)
target_classes = [3, 4, 5, 6, 7, 8, 9, 10]  # <-- классы для генерации

# Максимальный размер для синтетики на класс — ограничиваем, например 1000
MAX_SYNTHETIC_SAMPLES = 1000

# Подсчет сколько генерировать для каждого класса
max_count = train_df['Label'].value_counts().max()
target_counts = {}
for cls in target_classes:
    if cls in train_df['Label'].unique():
        current_count = len(train_df[train_df['Label'] == cls])
        needed = max_count - current_count
        # Ограничиваем максимум генерации
        gen_count = min(needed, MAX_SYNTHETIC_SAMPLES) if needed > 0 else 0
        target_counts[cls] = gen_count

# Используем новый Metadata класс
metadata = Metadata()
metadata.detect_from_dataframe(data=train_df)

augmented_frames = []


for label_value, gen_count in target_counts.items():
    df_class = train_df[train_df['Label'] == label_value]

    if len(df_class) < 50:
        print(f"Пропускаем класс {label_value}, мало данных ({len(df_class)})")
        continue
    if gen_count == 0:
        print(f"Класс {label_value} уже сбалансирован или превышен, генерировать не нужно")
        continue

    print(f"Обучаем CTGAN для класса {label_value} ({len(df_class)} реальных), генерируем {gen_count}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_class)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df_class)

    new_samples = synthesizer.sample(num_rows=gen_count)
    augmented_frames.append(new_samples)


# Объединяем все синтетические данные с оригинальным train
if augmented_frames:
    gan_data = pd.concat(augmented_frames, ignore_index=True)
    new_train = pd.concat([train_df, gan_data], ignore_index=True)
else:
    new_train = train_df

print("Размер нового train:", new_train.shape)
print(new_train['Label'].value_counts())

# Сохраняем итог
new_train.to_csv('datasets/gan_augmented_train.csv', index=False)
