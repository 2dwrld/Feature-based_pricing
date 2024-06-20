import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.data_loading import load_data
from scripts.preprocessing import preprocess_data
from scripts.feature_engineering import create_features
from scripts.hyperparameter_tuning import tune_hyperparameters
from scripts.model import predict_price_range
from sklearn.impute import KNNImputer
from clearml import Task, Logger

# Initialize ClearML task
task = Task.init(project_name="Car Price Prediction", task_name="Streamlit App")


def main():
    st.title("Car Price Prediction")

    file_path = 'data/imports-85.data'
    param_file = 'best_params.json'
    df = load_data(file_path)
    df = preprocess_data(df)
    df = impute_nan_knn(df)
    X, y, cat_features = create_features(df)
    best_model = tune_hyperparameters(X, y, param_file)

    st.sidebar.header("Car Details Input")
    input_df, input_categoricals = get_user_input(df, X)

    if st.button("Predict Price Range"):
        st.write("Input data passed to predict_price_range:\n", input_df)  # Debug print
        price_prediction_low, price_prediction_high, calibrated_price_low, calibrated_price_high = predict_price_range(
            best_model, input_df, X, y, input_categoricals)

        st.subheader("Predicted Price Range")
        st.write(
            f"Прогнозируемая цена автомобиля (стекинг): от {int(price_prediction_low)} до {int(price_prediction_high)}")
        st.write(f"Откалиброванная цена автомобиля: от {int(calibrated_price_low)} до {int(calibrated_price_high)}")

        # Log metrics to ClearML
        task.get_logger().report_scalar("price_prediction", "low", int(price_prediction_low), 0)
        task.get_logger().report_scalar("price_prediction", "high", int(price_prediction_high), 0)
        task.get_logger().report_scalar("calibrated_price", "low", int(calibrated_price_low), 0)
        task.get_logger().report_scalar("calibrated_price", "high", int(calibrated_price_high), 0)

    if st.button("Show Correlation Matrix"):
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
        "normalized-losses", "symboling", "wheel-base", "length", "width", "height", "curb-weight", "engine-size",
        "compression-ratio", "city-mpg", "highway-mpg", "price"
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

    # Process mandatory fields
    for column in mandatory_fields:
        if column in categorical_options:
            value = st.sidebar.selectbox(f"{column}", categorical_options[column])
            input_data.append(value)
            input_categoricals[column] = value
        else:
            value = st.sidebar.number_input(f"{column}", min_value=0.0)
            input_data.append(value)
            input_categoricals[column] = value

    input_df = pd.DataFrame([input_data], columns=mandatory_fields)
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
    st.pyplot(plt)
    plt.savefig('correlation_matrix.png')
    task.get_logger().report_image("Correlation Matrix", "Correlation Matrix", iteration=0, image_path='correlation_matrix.png')


if __name__ == "__main__":
    main()
