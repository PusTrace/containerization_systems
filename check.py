import pandas as pd

data = pd.read_csv('test_data.csv', sep=';')
print(data.shape)       # (строки, колонки)
print(data.columns)     # список всех колонок
print(data.head(5))     # первые 5 строк, чтобы прикинуть структуру
print(data.info())      # инфо о типах данных и пропусках
print(data['Label'].value_counts())

print(data.isnull().sum())  # сколько пропущенных значений в каждом столбце
import matplotlib.pyplot as plt

data['Label'].value_counts().plot(kind='bar')
plt.title('Распределение меток в датасете')
plt.xlabel('Метки')
plt.ylabel('Количество')
plt.show()

import seaborn as sns

sns.histplot(data=data, x='Flow Duration', bins=30, kde=True)
plt.title('Распределение Flow Duration')
plt.show()
