import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Замена символа '?' на NaN
    df.replace('?', np.nan, inplace=True)

    # Определение числовых колонок
    numeric_columns = ["normalized-losses", "wheel-base", "length", "width", "height",
                       "curb-weight", "engine-size", "bore", "stroke",
                       "compression-ratio", "horsepower", "peak-rpm",
                       "city-mpg", "highway-mpg", "price"]

    # Преобразование значений в числовой формат
    for column in numeric_columns:
        df[column] = df[column].astype(float)

    # Заполнение пропущенных значений медианой для числовых колонок
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Заполнение пропущенных значений модой для категориальных колонок
    df.fillna(df.mode().iloc[0], inplace=True)

    # Стандартизация числовых признаков (кроме цены)
    scaler = StandardScaler()
    df[numeric_columns[:-1]] = scaler.fit_transform(df[numeric_columns[:-1]])

    return df
