import pandas as pd

def check_columns():
    for column in columns:
        # считаем количество уникальных значений столбца
        x = data[column].value_counts()

        # проверяем самый часто встречающийся класс
        # если значение больше 90%, то отмечаем столбец как не годный
        if x.values[0] > (len(data[column]) * 0.9):
            bad_columns.append(column)

    print(bad_columns)

if __name__ == "__main__":
    # Загрузка данных из CSV
    data = pd.read_csv(r'balanced_output.csv', encoding='latin1', sep=';')

    # получаем названия столбцов
    columns = data.columns

    # пустой массив для "плохих" столбцов
    bad_columns = []

    check_columns()
