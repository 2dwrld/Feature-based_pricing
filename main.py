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


def main():
    file_path = 'data/imports-85.data'
    param_file = 'best_params.json'
    df = load_data(file_path)
    df = preprocess_data(df)
    df = impute_nan_knn(df)
    X, y, cat_features = create_features(df)
    best_model = tune_hyperparameters(X, y, param_file)

    input_df, input_categoricals = get_user_input(df, X)

    print("Input data passed to predict_price_range:\n", input_df)  # Debug print
    price_prediction_low, price_prediction_high, calibrated_price_low, calibrated_price_high = predict_price_range(
        best_model, input_df, X, y, input_categoricals)

    print(f"Прогнозируемая цена автомобиля (стекинг): от {int(price_prediction_low)} до {int(price_prediction_high)}")
    print(f"Откалиброванная цена автомобиля: от {int(calibrated_price_low)} до {int(calibrated_price_high)}")
    plot_correlation_matrix(df)


def impute_nan_knn(df):
    numeric_columns_with_nan = df.columns[df.isnull().any()]
    df_imputed = df.copy()
    if not numeric_columns_with_nan.empty:
        imputer = KNNImputer()
        df_imputed[numeric_columns_with_nan] = imputer.fit_transform(df[numeric_columns_with_nan])
    return df_imputed


def get_user_input(df, X):
    input_data = []
    input_categoricals = {}
    mandatory_fields = [
        "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "engine-type", "fuel-system", "num-of-cylinders", "bore", "stroke", "horsepower", "peak-rpm",
        "normalized-losses"
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
    for column in mandatory_fields:
        if column in categorical_options:
            print(f"Possible values for {column}: {categorical_options[column]}")
            value = input(f"{column}: ")
            while value not in categorical_options[column]:
                print(f"Invalid value. Possible values for {column}: {categorical_options[column]}")
                value = input(f"{column}: ")
            input_data.append(value)
            input_categoricals[column] = value
        else:
            value = float(input(f"{column}: "))
            input_data.append(value)
            input_categoricals[column] = value
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
                input_categoricals[column] = value
            else:
                value = input(f"{column} (optional): ")
                if value == '':
                    value = df[column].median()
                else:
                    value = float(value)
                input_data.append(value)
                input_categoricals[column] = value
    input_df = pd.DataFrame([input_data], columns=df.columns)
    input_df = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    return input_df, input_categoricals


def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[float, int])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Correlation matrix saved as 'correlation_matrix.png'")


def build_stacked_model(base_models, meta_model, X, y):
    stack = StackingRegressor(regressors=base_models, meta_regressor=meta_model)
    stack.fit(X, y)
    return stack


if __name__ == "__main__":
    main()
