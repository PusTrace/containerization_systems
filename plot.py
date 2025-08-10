import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

dataset = pd.read_csv('gan_smote/gan_augmented_train.csv')
X_train = dataset.drop(columns=['Label'])
y_train = dataset['Label']

# Допустим, X_train — твои признаки, y_train — метки (числа или категории)
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_train)

plt.figure(figsize=(12, 8))

# Получаем уникальные метки и цвета для них
unique_labels = np.unique(y_train)
colors = plt.cm.get_cmap('tab10', len(unique_labels))

for i, label in enumerate(unique_labels):
    idx = (y_train == label)
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], 
                color=colors(i), label=f'Класс {label}', alpha=0.7, s=30)

plt.legend()
plt.title('t-SNE визуализация с цветами по меткам')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.show()
