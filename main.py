import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scripts.data_loading import load_data
from scripts.hyperparameter_tuning import tune_hyperparameters
from scripts.preprocessing import preprocess_data
from scripts.feature_engineering import create_features
from scripts.model import predict_price_range, train_model
from sklearn.impute import KNNImputer
from mlxtend.regressor import StackingRegressor
from clearml import Task

# Инициализация задачи ClearML для отслеживания экспериментов и логирования
task = Task.init(project_name="Car Price Prediction", task_name="Main Task")


def main():
    # Пути к файлам
    file_path = 'data/imports-85.data'
    param_file = 'best_params.json'

    # Загрузка и предварительная обработка данных
    df = load_data(file_path)
    df = preprocess_data(df)

    # Импутация пропущенных значений с помощью KNN
    df = impute_nan_knn(df)

    # Создание признаков и целевой переменной
    X, y, cat_features = create_features(df)

    # Подбор гиперпараметров и получение лучшей модели
    best_model = tune_hyperparameters(X, y, param_file)

    # Получение данных от пользователя для предсказания
    input_df, input_categoricals = get_user_input(df, X)

    # Печать входных данных для целей отладки
    print("Input data passed to predict_price_range:\n", input_df)

    # Прогнозирование диапазона цен с использованием лучшей модели
    price_prediction_low, price_prediction_high, calibrated_price_low, calibrated_price_high = predict_price_range(
        best_model, input_df, X, y, input_categoricals)

    # Печать предсказанного диапазона цен
    print(f"Прогнозируемая цена автомобиля (стекинг): от {int(price_prediction_low)} до {int(price_prediction_high)}")
    print(f"Откалиброванная цена автомобиля: от {int(calibrated_price_low)} до {int(calibrated_price_high)}")

    # Построение и сохранение матрицы корреляций
    plot_correlation_matrix(df)

    # Логирование метрик в ClearML
    task.get_logger().report_scalar("price_prediction", "low", int(price_prediction_low), 0)
    task.get_logger().report_scalar("price_prediction", "high", int(price_prediction_high), 0)
    task.get_logger().report_scalar("calibrated_price", "low", int(calibrated_price_low), 0)
    task.get_logger().report_scalar("calibrated_price", "high", int(calibrated_price_high), 0)


def impute_nan_knn(df):
    # Определение колонок с пропущенными значениями
    numeric_columns_with_nan = df.columns[df.isnull().any()]
    df_imputed = df.copy()

    # Если есть колонки с пропущенными значениями, выполняем их импутацию с помощью KNN
    if not numeric_columns_with_nan.empty:
        imputer = KNNImputer()
        df_imputed[numeric_columns_with_nan] = imputer.fit_transform(df[numeric_columns_with_nan])
    return df_imputed


def get_user_input(df, X):
    # Инициализация списка для входных данных и словаря для категориальных данных
    input_data = []
    input_categoricals = {}

    # Обязательные поля для ввода
    mandatory_fields = [
        "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "engine-type", "fuel-system", "num-of-cylinders", "bore", "stroke", "horsepower", "peak-rpm",
        "normalized-losses", "symboling", "wheel-base", "length", "width", "height", "curb-weight", "engine-size",
        "compression-ratio", "city-mpg", "highway-mpg", "price"
    ]

    # Определение возможных значений для категориальных полей
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

    # Обработка обязательных полей
    for column in mandatory_fields:
        if column in categorical_options:
            # Если поле категориальное, запрашиваем ввод значения из возможных вариантов
            value = input(f"{column}: ")
            while value not in categorical_options[column]:
                print(f"Invalid value. Possible values for {column}: {categorical_options[column]}")
                value = input(f"{column}: ")
            input_data.append(value)
            input_categoricals[column] = value
        else:
            # Если поле числовое, запрашиваем ввод числового значения
            value = float(input(f"{column}: "))
            input_data.append(value)
            input_categoricals[column] = value

    # Создание DataFrame из введенных данных и преобразование категориальных переменных в dummy переменные
    input_df = pd.DataFrame([input_data], columns=mandatory_fields)
    input_df = pd.get_dummies(input_df)

    # Добавление отсутствующих колонок
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Реиндексация DataFrame в соответствии с колонками исходных данных
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return input_df, input_categoricals


def plot_correlation_matrix(df):
    # Выбор только числовых колонок для построения матрицы корреляций
    numeric_df = df.select_dtypes(include=[float, int])
    corr = numeric_df.corr()

    # Построение и сохранение матрицы корреляций
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved as 'correlation_matrix.png'")

    # Логирование матрицы корреляций в ClearML
    task.get_logger().report_image("Correlation Matrix", "Correlation Matrix", iteration=0,
                                   image_path='correlation_matrix.png')


if __name__ == "__main__":
    main()
