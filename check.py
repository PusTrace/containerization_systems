import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_column_existence(data):
    print(data.shape)       # (строки, колонки)
    print(data.columns)     # список всех колонок
    print(data.head(5))     # первые 5 строк, чтобы прикинуть структуру
    print(data.info())      # инфо о типах данных и пропусках
    print(data['Label'].value_counts())


def plot_label_distribution(data):
    plt.figure(figsize=(10, 5))
    data['Label'].value_counts().plot(kind='bar')
    plt.title('Распределение меток в датасете')
    plt.xlabel('Метки')
    plt.ylabel('Количество')
    plt.show()



def plot_flow_duration_distribution(data):
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Flow Duration'], bins=30, kde=True)
    plt.title('Распределение Flow Duration')
    plt.xlabel('Flow Duration')
    plt.ylabel('Частота')
    plt.show()


def plot_zero_ratio_by_column(data, top_n=None):
    # Считаем долю нулей в каждом столбце
    zero_ratio = ((data == 0).sum() / len(data)).sort_values(ascending=False)

    # Если хотим показать только top_n столбцов
    if top_n:
        zero_ratio = zero_ratio.head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=zero_ratio.index, y=zero_ratio.values, palette="viridis", legend=False)
    plt.title("Доля нулей по колонкам")
    plt.ylabel("Доля нулей")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return zero_ratio


if __name__ == "__main__":
    bout = 'balanced_output.csv'
    out = 'output.csv'

    data = pd.read_csv('datasets/gan_augmented_train.csv', sep=',')
    check_column_existence(data)
    # plot_label_distribution(data)
    # ratios = plot_zero_ratio_by_column(data, top_n=20)
    # print(ratios)

