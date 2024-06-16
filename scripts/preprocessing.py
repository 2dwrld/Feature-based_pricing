# scripts/preprocessing.py

import numpy as np


def preprocess_data(df):
    df.replace('?', np.nan, inplace=True)

    numeric_columns = ["normalized-losses", "wheel-base", "length", "width", "height",
                       "curb-weight", "engine-size", "bore", "stroke",
                       "compression-ratio", "horsepower", "peak-rpm",
                       "city-mpg", "highway-mpg", "price"]

    for column in numeric_columns:
        df[column] = df[column].astype(float)

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    return df
