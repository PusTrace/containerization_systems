import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight
import numpy as np

# 1. Загрузка данных с sep=';'
train_df = pd.read_csv('gan_smote/gan_augmented_train.csv', sep=',')
val_df = pd.read_csv('gan_smote/validation.csv', sep=',')
test_df = pd.read_csv('gan_smote/test.csv', sep=',')


# 2. Разделяем признаки и метки
X_train = train_df.drop(columns=['Label'])
y_train = train_df['Label']

X_val = val_df.drop(columns=['Label'])
y_val = val_df['Label']

X_test = test_df.drop(columns=['Label'])
y_test = test_df['Label']

# 3. Создаем объекты Pool
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)


# 3. Вычисляем веса классов
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Преобразуем список весов в словарь, который нужен для CatBoost
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Увеличиваем вес классов 3-11 в два раза
for class_label in range(3, 12):
    if class_label in class_weights_dict:
        class_weights_dict[class_label] *= 2  # Увеличиваем вес класса в 2 раз

# 4. Создаем объекты Pool с указанием весов
train_pool = Pool(X_train, y_train, weight=[class_weights_dict[label] for label in y_train])
val_pool = Pool(X_val, y_val)

# 4. Инициализация модели CatBoost (пример без Randomized Search, для теста)
model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    task_type='GPU',
    devices='0',
    random_state=42,
    verbose=10
)

# 5. Обучение с использованием валидационного сета
model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,
    early_stopping_rounds=50
)

# 6. Предсказания на тестовом сете
y_pred = model.predict(X_test).flatten()

# 7. Метрики
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

print("\nОтчет классификации:\n", classification_report(y_test, y_pred))

# 8. Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Матрица ошибок')
plt.show()
