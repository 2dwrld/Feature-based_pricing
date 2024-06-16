# scripts/model.py

from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_price(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]
