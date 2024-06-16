import numpy as np
import pandas as pd

from scripts.data_loading import load_data
from scripts.preprocessing import preprocess_data
from scripts.feature_engineering import create_features
from scripts.model import train_model, predict_price


def main():
    # Путь к файлу с данными
    file_path = 'data/imports-85.data'

    # Шаг 1: Загрузка данных
    df = load_data(file_path)
    print("Данные загружены")

    # Шаг 2: Первичная обработка данных
    df = preprocess_data(df)
    print("Данные обработаны")

    # Шаг 3: Формирование признаков
    X, y = create_features(df)
    print("Признаки созданы")

    # Шаг 4: Обучение модели
    model = train_model(X, y)
    print("Модель обучена")

    # Шаг 5: Прогнозирование цены на основе пользовательского ввода
    input_data = get_user_input(df)
    price_prediction = predict_price(model, input_data)
    print(f"Прогнозируемая цена автомобиля: {price_prediction}")


def get_user_input(df):
    input_data = []

    # Определяем возможные значения для категориальных переменных
    categorical_options = {
        "make": df["make"].unique().tolist(),
        "fuel-type": df["fuel-type"].unique().tolist(),
        "aspiration": df["aspiration"].unique().tolist(),
        "num-of-doors": df["num-of-doors"].unique().tolist(),
        "body-style": df["body-style"].unique().tolist(),
        "drive-wheels": df["drive-wheels"].unique().tolist(),
        "engine-location": df["engine-location"].unique().tolist(),
        "engine-type": df["engine-type"].unique().tolist(),
        "num-of-cylinders": df["num-of-cylinders"].unique().tolist(),
        "fuel-system": df["fuel-system"].unique().tolist(),
    }

    # Запрашиваем у пользователя значения для всех столбцов
    print("Введите значения для следующих характеристик автомобиля:")
    for column in df.columns:
        if column in categorical_options:
            print(f"Возможные значения для {column}: {categorical_options[column]}")
            value = input(f"{column}: ")
            while value not in categorical_options[column]:
                print(f"Недопустимое значение. Возможные значения для {column}: {categorical_options[column]}")
                value = input(f"{column}: ")
            input_data.append(value)
        else:
            value = float(input(f"{column}: "))
            input_data.append(value)

    # Преобразуем категориальные признаки в dummy-переменные
    input_df = pd.DataFrame([input_data], columns=df.columns)
    input_df = pd.get_dummies(input_df)

    # Дополняем недостающие dummy-переменные нулями
    for col in df.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_data = input_df.values[0]
    return input_data


if __name__ == "__main__":
    main()
