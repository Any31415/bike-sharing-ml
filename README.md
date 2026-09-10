# Bike Sharing ML — прогноз спроса на велопрокат

Лабораторная работа №1 «Классический жизненный цикл разработки моделей
машинного обучения». Вариант 2 — Kaggle Bike Sharing Demand.

## Что делает сервис

FastAPI-сервис принимает параметры поездки (погода, время, день недели)
и возвращает прогноз количества аренд велосипедов (модель CatBoostRegressor).

## Быстрый запуск

Модель уже обучена и включена в Docker-образ — dvc pull не требуется.

```bash
docker-compose up -d
```

Открыть http://localhost:8000/docs и проверить POST /predict, например:

```json
{
  "season": 3, "holiday": 0, "workingday": 1, "weather": 1,
  "temp": 25, "humidity": 55, "windspeed": 8,
  "hour": 8, "dayofweek": 2, "month": 7
}
```

## Структура проекта

| Путь | Назначение |
|---|---|
| `src/train.py` | обучение модели по config.ini |
| `src/predict.py` | функция инференса |
| `src/api.py` | FastAPI-сервис |
| `tests/` | unit-тесты (pytest) |
| `notebooks/model.ipynb` | EDA и эксперименты |
| `config.ini` | гиперпараметры модели |
| `Dockerfile`, `docker-compose.yml` | контейнеризация |
| `.github/workflows/ci.yml` | CI: тесты + сборка/push образа |
| `.github/workflows/cd.yml` | CD: функциональные тесты контейнера |

## Данные и модель (DVC)

`data/`, `models/model.pkl` версионируются через DVC, remote — Google Drive.
Для локальной работы с полным репозиторием (обучение с нуля):

```bash
dvc pull
```

Для проверки уже готового сервиса это не нужно — модель уже внутри
Docker-образа на DockerHub.

## Ссылки

- Docker image: https://hub.docker.com/r/ani31415/bike-sharing-api