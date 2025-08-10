from sklearn.model_selection import train_test_split
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Загружаем данные
balanced_data = pd.read_csv('balanced_output.csv')

# Убираем нормальный трафик
balanced_data = balanced_data[balanced_data['label'] != 0].reset_index(drop=True)

# Объединяем редкие классы в "other"
balanced_data['Label'] = balanced_data['Label'].replace({10: 99, 5: 99, 9: 99})

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

# Классы для GAN
target_classes = [8, 6, 3, 4, 7, 99, 11]

# Определяем сколько нужно сгенерировать для баланса
max_count = train_df['Label'].value_counts().max()
target_counts = {cls: max_count - len(train_df[train_df['Label'] == cls])
                 for cls in target_classes if cls in train_df['Label'].unique()}

# Метаинформация для CTGAN
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train_df)

augmented_frames = []

for label_value, gen_count in target_counts.items():
    df_class = train_df[train_df['Label'] == label_value]

    if len(df_class) < 50:
        print(f"Пропускаем класс {label_value}, мало данных ({len(df_class)})")
        continue

    print(f"Обучаем CTGAN для класса {label_value} ({len(df_class)} реальных), генерируем {gen_count}")

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df_class)

    new_samples = synthesizer.sample(num_rows=gen_count)
    augmented_frames.append(new_samples)

# Объединяем всё
if augmented_frames:
    gan_data = pd.concat(augmented_frames, ignore_index=True)
    new_train = pd.concat([train_df, gan_data], ignore_index=True)
else:
    new_train = train_df

# Сохраняем итог
new_train.to_csv('gan_smote/gan_augmented_train.csv', index=False)

print("Размер нового train:", new_train.shape)
print(new_train['Label'].value_counts())
