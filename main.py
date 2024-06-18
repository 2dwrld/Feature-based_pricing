import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scripts.data_loading import load_data
from scripts.hyperparameter_tuning import tune_hyperparameters
from scripts.preprocessing import preprocess_data
from scripts.feature_engineering import create_features
from scripts.model import predict_price_range

from sklearn.impute import KNNImputer
from mlxtend.regressor import StackingRegressor


def main():
    # Путь к файлу с данными
    file_path = 'data/imports-85.data'

    # Шаг 1: Загрузка данных
    df = load_data(file_path)
    print("Данные загружены")

    # Шаг 2: Первичная обработка данных
    df = preprocess_data(df)
    print("Данные обработаны")

    # Шаг 2.5: Импутация пропущенных значений с использованием метода KNN
    df = impute_nan_knn(df)
    print("Пропущенные значения заполнены методом KNN")

    # Шаг 3: Формирование признаков
    X, y, cat_features = create_features(df)
    print("Признаки созданы")

    # Шаг 4: Подбор гиперпараметров для ансамблевой модели
    best_model = tune_hyperparameters(X, y)
    print("Гиперпараметры подобраны")

    # Шаг 5: Прогнозирование цены на основе пользовательского ввода
    input_data = get_user_input(df, X)
    price_prediction_low, price_prediction_high = predict_price_range(best_model, input_data, X, y)
    print(f"Прогнозируемая цена автомобиля (стекинг): от {price_prediction_low} до {price_prediction_high}")

    # Шаг 6: Построение корреляционной матрицы
    plot_correlation_matrix(df)


def impute_nan_knn(df):
    # Identify numeric columns with missing values
    numeric_columns_with_nan = df.columns[df.isnull().any()]

    # Copy dataframe to avoid modifying original
    df_imputed = df.copy()

    if not numeric_columns_with_nan.empty:
        # Initialize KNN imputer
        imputer = KNNImputer()

        # Fit and transform the data
        df_imputed[numeric_columns_with_nan] = imputer.fit_transform(df[numeric_columns_with_nan])

    return df_imputed


def get_user_input(df, X):
    input_data = []

    mandatory_fields = [
        "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "engine-size", "horsepower",
        "curb-weight", "city-mpg", "highway-mpg"
    ]

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

    # Request user input for mandatory fields
    print("Enter values for the following car characteristics:")
    for column in mandatory_fields:
        if column in categorical_options:
            print(f"Possible values for {column}: {categorical_options[column]}")
            value = input(f"{column}: ")
            while value not in categorical_options[column]:
                print(f"Invalid value. Possible values for {column}: {categorical_options[column]}")
                value = input(f"{column}: ")
            input_data.append(value)
        else:
            value = float(input(f"{column}: "))
            input_data.append(value)

    # Fill in missing values for optional fields with mode or median
    for column in df.columns:
        if column not in mandatory_fields:
            if column in categorical_options:
                print(f"Possible values for {column}: {categorical_options[column]}")
                value = input(f"{column} (optional): ")
                if value == '':
                    value = df[column].mode()[0]
                while value not in categorical_options[column]:
                    print(f"Invalid value. Possible values for {column}: {categorical_options[column]}")
                    value = input(f"{column}: ")
                input_data.append(value)
            else:
                value = input(f"{column} (optional): ")
                if value == '':
                    value = df[column].median()
                else:
                    value = float(value)
                input_data.append(value)

    # Convert categorical features to dummy variables
    input_df = pd.DataFrame([input_data], columns=df.columns)
    input_df = pd.get_dummies(input_df)

    # Add missing dummy variables with zeros
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Align the column order with the training set
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    return input_df


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[float, int])  # Select only numeric columns
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')  # Save the plot to a file
    print("Correlation matrix saved as 'correlation_matrix.png'")


def build_stacked_model(base_models, meta_model, X, y):
    stack = StackingRegressor(regressors=base_models, meta_regressor=meta_model)
    stack.fit(X, y)
    return stack


if __name__ == "__main__":
    main()
