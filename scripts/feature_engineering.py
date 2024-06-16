# scripts/feature_engineering.py

import pandas as pd


def create_features(df):
    categorical_columns = ["make", "fuel-type", "aspiration", "num-of-doors",
                           "body-style", "drive-wheels", "engine-location",
                           "engine-type", "num-of-cylinders", "fuel-system"]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    X = df.drop("price", axis=1)
    y = df["price"]

    return X, y
