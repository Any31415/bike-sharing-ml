import joblib
import numpy as np
import pandas as pd


def load_model(model_path='models/model.pkl'):
    return joblib.load(model_path)


def predict(model, features: dict) -> float:
    """
    features — словарь вида:
    {
        'season': 1, 'holiday': 0, 'workingday': 1, 'weather': 1,
        'temp': 20.5, 'humidity': 60, 'windspeed': 10.0,
        'hour': 8, 'dayofweek': 2, 'month': 6
    }
    """
    X = pd.DataFrame([features])
    pred_log = model.predict(X)[0]
    pred = np.expm1(pred_log)  # обратно из логарифма в реальное число аренд
    return round(float(pred), 2)


if __name__ == '__main__':
    model = load_model()
    example = {
        'season': 1, 'holiday': 0, 'workingday': 1, 'weather': 1,
        'temp': 20.5, 'humidity': 60, 'windspeed': 10.0,
        'hour': 8, 'dayofweek': 2, 'month': 6
    }
    result = predict(model, example)
    print(f'Predicted count: {result}')