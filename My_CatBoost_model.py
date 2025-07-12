import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
df = pd.read_csv('balanced_output.csv')
df = df.drop(columns=['Flow ID'])

X = df.drop(columns=['Label'])
y = df['Label']

# 2. Разделение на train/val/test (70/10/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
)

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# 3. Инициализация модели для Randomized Search с GPU
model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    task_type='GPU',
    devices='0',  # номер GPU, обычно 0
    random_state=42,
    verbose=0
)

# 4. Простая Randomized Search
param_distributions = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [500, 1000, 1500],
    'border_count': [32, 50, 100]
}

random_search_result = model.randomized_search(
    param_distributions,
    train_pool,
    n_iter=20,
    cv=3,
    stratified=True,
    refit=True,
    plot=False,
    verbose=True
)

best_params = random_search_result['params']
print("Лучшие параметры:", best_params)

# 5. Обучение финальной модели с лучшими параметрами
final_model = CatBoostClassifier(
    **best_params,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    task_type='GPU',
    devices='0',
    random_state=42,
    verbose=10
)

final_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,
    early_stopping_rounds=50
)

# 6. Сохранение модели
final_model.save_model("model.cbm")

# 7. Предсказание и метрики
y_pred = final_model.predict(X_test).flatten()

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
