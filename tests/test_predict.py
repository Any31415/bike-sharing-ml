from src.predict import load_model, predict

model = load_model()


def test_predict_returns_number():
    features = {
        'season': 1, 'holiday': 0, 'workingday': 1, 'weather': 1,
        'temp': 20.5, 'humidity': 60, 'windspeed': 10.0,
        'hour': 8, 'dayofweek': 2, 'month': 6
    }
    result = predict(model, features)
    assert isinstance(result, float)


def test_predict_is_non_negative():
    features = {
        'season': 1, 'holiday': 0, 'workingday': 1, 'weather': 1,
        'temp': 20.5, 'humidity': 60, 'windspeed': 10.0,
        'hour': 8, 'dayofweek': 2, 'month': 6
    }
    result = predict(model, features)
    assert result >= 0