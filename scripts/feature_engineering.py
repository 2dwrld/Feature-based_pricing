import pandas as pd

def create_features(df):
    categorical_columns = [
        "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "engine-type", "num-of-cylinders",
        "fuel-system"
    ]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    X = df.drop("price", axis=1)
    y = df["price"]

    # Get the indices of the categorical features in the new dataframe
    cat_feature_indices = [X.columns.get_loc(col) for col in X.columns if any(cat_col in col for cat_col in categorical_columns)]

    return X, y, cat_feature_indices
