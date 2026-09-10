import configparser
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import joblib


def load_config(path='config.ini'):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def prepare_features(df, feature_columns):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    X = df[feature_columns]
    y = np.log1p(df['count'])
    return X, y


def train_model(cfg):
    feature_columns = cfg['features']['columns'].split(',')

    df = pd.read_csv(cfg['data']['train_path'])
    X, y = prepare_features(df, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.getint('model', 'random_state')
    )

    model = CatBoostRegressor(
        iterations=cfg.getint('model', 'iterations'),
        learning_rate=cfg.getfloat('model', 'learning_rate'),
        depth=cfg.getint('model', 'depth'),
        random_state=cfg.getint('model', 'random_state'),
        verbose=100
    )
    model.fit(X_train, y_train)

    joblib.dump(model, cfg['data']['model_path'])
    print(f"Model saved to {cfg['data']['model_path']}")

    return model


if __name__ == '__main__':
    config = load_config()
    train_model(config)