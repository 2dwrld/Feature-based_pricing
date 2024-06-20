import numpy as np
from catboost import CatBoostRegressor
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
    predictions = model.predict(X)
    std_dev = np.std(predictions - y)
    prediction = model.predict(input_data)[0]
    price_prediction_low = prediction - std_dev
    price_prediction_high = prediction + std_dev

    # Debug prints for original predictions
    print(f"Model prediction: {prediction}")
    print(f"Predicted price range: {price_prediction_low} - {price_prediction_high}")

    # Fetch market prices from auto.ru
    print("Fetching car listings with the following parameters:")
    print(
        f"make: {input_categoricals['make']}, body_style: {input_categoricals['body-style']}, engine_type: {input_categoricals.get('engine-type', 'N/A')}, drive_type: {input_categoricals['drive-wheels']}")
    car_listings = get_car_listings(
        make=input_categoricals['make'],
        body_style=input_categoricals['body-style'],
        drive_type=input_categoricals['drive-wheels'],
        engine_size=input_categoricals.get('engine-size', None),
        horsepower=input_categoricals.get('horsepower', None)
    )

    print(f"Fetched car listings: {car_listings}")

    if not car_listings:
        print("No car listings found, using model's predicted range.")
        return price_prediction_low, price_prediction_high, price_prediction_low, price_prediction_high

    exchange_rate = get_usd_to_rub_exchange_rate()
    print(f"Current USD to RUB exchange rate: {exchange_rate}")

    # Convert market prices to USD
    market_prices_usd = [price / exchange_rate for price in car_listings]
    print(f"Market prices in USD: {market_prices_usd}")

    # Normalize prediction based on market data
    market_price_mean = np.mean(market_prices_usd)
    std_dev_market = np.std(market_prices_usd)
    calibrated_price_low = market_price_mean - std_dev_market
    calibrated_price_high = market_price_mean + std_dev_market

    # Debug prints for calibrated prices
    print(f"Market price mean: {market_price_mean}")
    print(f"Market price standard deviation: {std_dev_market}")
    print(f"Calibrated price range: {calibrated_price_low} - {calibrated_price_high}")

    return price_prediction_low, price_prediction_high, calibrated_price_low, calibrated_price_high