from catboost import CatBoostRegressor
import numpy as np
from mlxtend.regressor import StackingRegressor
from scripts.auto_ru_scraper import get_car_listings
from scripts.currency_conversion import get_usd_to_rub_exchange_rate


class CustomStackingRegressor(StackingRegressor):
    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith('regressors__'):
                _, idx, param = key.split('__', 2)
                self.regressors[int(idx)].set_params(**{param: value})
            elif key.startswith('meta_regressor__'):
                _, param = key.split('__', 1)
                self.meta_regressor.set_params(**{param: value})
            else:
                setattr(self, key, value)
        return self


def train_model(X, y, cat_features):
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=0)
    model.fit(X, y)
    return model


def predict_price_range(model, input_data, X, y, input_categoricals):
    # Original model predictions
    print("Input data passed to predict_price_range:\n", input_data)  # Debug print
    predictions = model.predict(X)
    std_dev = np.std(predictions - y)
    prediction = model.predict(input_data)[0]
    price_prediction_low = prediction - std_dev
    price_prediction_high = prediction + std_dev

    # Fetch market prices from auto.ru
    print("Fetching car listings with the following parameters:")
    print(
        f"make: {input_categoricals['make']}, body_style: {input_categoricals['body-style']}, engine_type: {input_categoricals['engine-type']}, drive_type: {input_categoricals['drive-wheels']}")
    car_listings = get_car_listings(make=input_categoricals['make'], body_style=input_categoricals['body-style'],
                                    engine_type=input_categoricals.get('engine-type', ''),
                                    drive_type=input_categoricals['drive-wheels'])
    exchange_rate = get_usd_to_rub_exchange_rate()

    # Convert market prices to USD
    market_prices_usd = [price / exchange_rate for price in car_listings]

    # Normalize prediction based on market data
    if market_prices_usd:
        market_price_mean = np.mean(market_prices_usd)
        calibrated_price_low = market_price_mean * 0.9
        calibrated_price_high = market_price_mean * 1.1
    else:
        calibrated_price_low, calibrated_price_high = price_prediction_low, price_prediction_high

    return price_prediction_low, price_prediction_high, calibrated_price_low, calibrated_price_high
