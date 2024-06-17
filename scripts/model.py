from catboost import CatBoostRegressor
import numpy as np

def train_model(X, y, cat_features):
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=0)
    model.fit(X, y)
    return model

def predict_price_range(model, input_data, X, y):
    predictions = model.predict(X)
    std_dev = np.std(predictions - y)  # Adjusted standard deviation calculation
    prediction = model.predict([input_data])[0]
    print(prediction)
    print(f"y - {y}")
    price_prediction_low = prediction - std_dev
    price_prediction_high = prediction + std_dev

    return price_prediction_low, price_prediction_high
