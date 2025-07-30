import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных с sep=';'
train_df = pd.read_csv('train.csv', sep=';')
val_df = pd.read_csv('valid.csv', sep=';')
test_df = pd.read_csv('test_data.csv', sep=';')

# Удаляем лишний столбец, если он есть, например Flow ID
for df in [train_df, val_df, test_df]:
    if 'Flow ID' in df.columns:
        df.drop(columns=['Flow ID'], inplace=True)

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
