from catboost import CatBoostRegressor
import numpy as np
from mlxtend.regressor import StackingRegressor


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


def predict_price_range(model, input_data, X, y):
    predictions = model.predict(X)
    std_dev = np.std(predictions - y)
    prediction = model.predict(input_data)[0]
    price_prediction_low = prediction - std_dev
    price_prediction_high = prediction + std_dev

    return price_prediction_low, price_prediction_high
