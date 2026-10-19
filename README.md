# Credit Scoring MLOps Project

[![CI](https://github.com/daryashaporewa-netizen/credit-scoring-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/daryashaporewa-netizen/credit-scoring-mlops/actions/workflows/ci.yml)

Учебный end-to-end MLOps-проект для прогнозирования дефолта клиента кредитной карты.

Проект выполнен в рамках курса **«Автоматизация процессов разработки и тестирования моделей машинного обучения»**.

Цель проекта — построить не только модель бинарной классификации, но и воспроизводимый ML workflow вокруг неё: от проверки исходных данных и экспериментов до версионирования, автоматического тестирования, REST API, Docker и базового мониторинга data drift.

---

## Основные результаты

- использован датасет на **30 000 наблюдений**;
- train: **24 000** наблюдений;
- test: **6 000** наблюдений;
- проведено **5 модельных экспериментов**;
- лучший ROC-AUC: **≈ 0.779**;
- реализована валидация данных с **Great Expectations**;
- реализован единый `sklearn.Pipeline`;
- выполнен подбор гиперпараметров;
- эксперименты отслеживаются через **MLflow**;
- данные и ML-артефакты управляются через **DVC**;
- ML pipeline воспроизводится через `dvc repro`;
- реализованы unit-тесты;
- настроены **Black** и **Flake8**;
- настроен **GitHub Actions CI**;
- реализован FastAPI endpoint `POST /predict`;
- API протестирован локально и внутри Docker;
- реализован демонстрационный data drift monitoring через **PSI**;
- 100 тестовых наблюдений успешно обработаны работающим inference API.

### Финальная модель

В качестве финальной модели проекта используется **Gradient Boosting Tuned**.

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.779 |
| Precision | 0.665 |
| Recall | 0.368 |
| F1 | 0.474 |

Balanced Logistic Regression при этом показала более высокий Recall и F1, что демонстрирует важный для кредитного скоринга компромисс между различными типами ошибок.

---

## Содержание

- [Архитектура](#архитектура)
- [Быстрый запуск](#быстрый-запуск)
- [Данные](#данные)
- [Валидация данных](#валидация-данных)
- [Feature engineering](#feature-engineering)
- [ML Pipeline](#ml-pipeline)
- [Эксперименты и выбор модели](#эксперименты-и-выбор-модели)
- [MLflow](#mlflow)
- [DVC](#dvc)
- [Артефакты pipeline](#артефакты-pipeline)
- [Тестирование и качество кода](#тестирование-и-качество-кода)
- [Continuous Integration](#continuous-integration)
- [FastAPI](#fastapi)
- [Docker](#docker)
- [Data Drift Monitoring](#data-drift-monitoring)
- [Структура проекта](#структура-проекта)
- [Воспроизводимость](#воспроизводимость)
- [Ограничения](#ограничения)
- [Технологии](#технологии)

---

## Архитектура

```text
                    Raw dataset
                         │
                         ▼
              Great Expectations
                         │
                         ▼
                 Feature engineering
                         │
                         ▼
                 Train / Test split
                         │
                         ▼
                Model experiments
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
            MLflow            Metrics / ROC
               │                   │
               └─────────┬─────────┘
                         ▼
                   Final model
                         │
                         ▼
                    DVC pipeline
                         │
                         ▼
                  Tests + CI
                         │
                         ▼
                     FastAPI
                         │
                         ▼
                      Docker
                         │
                         ▼
               PSI monitoring demo
```

---

## Быстрый запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/daryashaporewa-netizen/credit-scoring-mlops.git
cd credit-scoring-mlops
```

### 2. Создать виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate
```

Проект протестирован на **Python 3.12**.

Для Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Установить зависимости

```bash
python -m pip install -r requirements.txt
```

Версии Python-зависимостей зафиксированы в `requirements.txt`.

### 4. Добавить исходный датасет

Используется датасет **Default of Credit Card Clients** из UCI Machine Learning Repository.

Официальный источник:

```text
UCI Machine Learning Repository
Default of Credit Card Clients
Dataset ID: 350
DOI: 10.24432/C55S3H
```

Исходная UCI-версия содержит 30 000 наблюдений и 23 входных признака.
В проекте используется подготовленный CSV с целевой колонкой:

```text
default.payment.next.month
```

Файл должен находиться по пути:

```text
data/raw/UCI_Credit_Card.csv
```

DVC remote в учебной версии проекта не настроен, поэтому после свежего `git clone` исходный CSV необходимо добавить вручную.

### 5. Воспроизвести ML pipeline

```bash
dvc repro
```

DVC последовательно выполнит:

```text
raw data
   ↓
data validation
   ↓
feature engineering
   ↓
train/test split
   ↓
model training and experiments
   ↓
metrics and model artifacts
```

После успешного выполнения создаются, в частности:

```text
data/processed/train.csv
data/processed/test.csv
models/credit_default_model.pkl
experiment_results.csv
roc_curve.png
metrics.json
```

### 6. Запустить тесты

```bash
python -m pytest -v
```

На момент финальной локальной проверки:

```text
3 passed
```

Проверка форматирования и linting:

```bash
black --check src tests
flake8 src tests
```

### 7. Запустить FastAPI

```bash
uvicorn src.api.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### 8. Запустить через Docker

Перед сборкой Docker-образа должна существовать обученная модель:

```text
models/credit_default_model.pkl
```

Поэтому после свежего клонирования сначала необходимо добавить исходный CSV и выполнить:

```bash
dvc repro
```

После этого:

```bash
docker build -t credit-scoring-api .
docker run --rm -p 8000:8000 credit-scoring-api
```

Swagger UI внутри контейнера:

```text
http://127.0.0.1:8000/docs
```

---

## Данные

Используется датасет **Default of Credit Card Clients**.
Задача — бинарная классификация: прогноз дефолта клиента в следующем месяце.

Целевая переменная:

```text
default.payment.next.month
```

Интерпретация:

- `0` — дефолта в следующем месяце не произошло;
- `1` — дефолт произошёл.

Исходные признаки содержат информацию о:

- кредитном лимите;
- возрасте;
- поле;
- образовании;
- семейном положении;
- истории задержек платежей;
- суммах задолженности;
- предыдущих платежах.

После подготовки выполняется стратифицированное разделение:

```text
Train: 24000
Test: 6000
```

---

## Валидация данных

Для проверки качества исходных данных используется **Great Expectations**.

В pipeline контролируются:

- наличие ожидаемых колонок;
- типы данных;
- допустимые диапазоны;
- корректность критичных значений.

Отдельный unit-тест проверяет отрицательный сценарий.

В корректный пример данных намеренно подставляется:

```text
AGE = 150
```

После этого проверяется, что такие данные не проходят validation.
Таким образом, аномальные данные могут быть обнаружены до этапа подготовки и обучения модели.

---

## Feature engineering

В `src/features/build_features.py` создаются четыре дополнительных признака.

### `AVG_UTILIZATION`

Средний размер задолженности относительно кредитного лимита.

### `AVG_PAYMENT_RATIO`

Средний платёж относительно кредитного лимита.

### `MAX_DELAY`

Максимальное значение статуса задержки платежа за рассматриваемый период.

### `DELAY_MONTHS`

Количество месяцев с положительным статусом задержки.

После feature engineering из обучающих данных удаляется `ID`.

---

## ML Pipeline

Предобработка и классификатор объединены в единый `sklearn.Pipeline`.

Для обработки признаков используется `ColumnTransformer`.

### Числовые признаки

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler
```

### Категориальные признаки

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

Использование единого pipeline снижает риск рассинхронизации preprocessing между обучением и inference.

---

## Эксперименты и выбор модели

В проекте проведено пять модельных экспериментов.

| Модель | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.761 | 0.662 | 0.359 | 0.465 |
| Logistic Regression Balanced | 0.761 | 0.464 | 0.592 | 0.520 |
| Gradient Boosting Default | 0.779 | 0.666 | 0.365 | 0.471 |
| Gradient Boosting Tuned | 0.779 | 0.665 | 0.368 | 0.474 |
| Gradient Boosting Alternative | 0.773 | 0.658 | 0.356 | 0.462 |

Для Gradient Boosting выполнялся подбор гиперпараметров.

Лучшие найденные параметры:

```text
learning_rate = 0.05
max_depth = 3
n_estimators = 150
```

### Почему выбран Gradient Boosting Tuned

Основным ориентиром был ROC-AUC — способность модели ранжировать клиентов по уровню кредитного риска.

Gradient Boosting показал:

```text
ROC-AUC ≈ 0.779
```

При этом Balanced Logistic Regression получила:

```text
Recall = 0.592
F1 = 0.520
```

То есть Balanced Logistic Regression обнаруживает больше клиентов с дефолтом, но имеет более низкую Precision.
Эксперименты показывают, что выбор модели зависит от бизнес-задачи.
Если приоритет — качество ранжирования кредитного риска в целом, Gradient Boosting выглядит предпочтительнее по ROC-AUC.
Если особенно дорого пропустить потенциальный дефолт, модель с более высоким Recall может быть предпочтительнее.
В реальной кредитной системе выбор модели и decision threshold должен учитывать стоимость False Positive и False Negative ошибок.

---

## MLflow

Для experiment tracking используется **MLflow**.

Во время обучения логируются:

- параметры моделей;
- ROC-AUC;
- Precision;
- Recall;
- F1;
- model artifacts.

Эксперименты создаются при запуске:

```bash
python -m src.models.train
```

или полного DVC pipeline:

```bash
dvc repro
```

Запуск MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Интерфейс:

```text
http://127.0.0.1:5000
```

`mlflow.db` используется как локальный Tracking Store и не хранится в Git.

Поэтому история локальных MLflow runs не поставляется вместе с `git clone`, но может быть воспроизведена повторным запуском обучения.

---

## DVC

Для управления данными и ML-артефактами используется **DVC**.

Pipeline описан в:

```text
dvc.yaml
```

Основные стадии:

```text
prepare
   ↓
train
```

### Stage `prepare`

Выполняет:

- чтение исходного CSV;
- data validation;
- feature engineering;
- удаление `ID`;
- train/test split;
- сохранение обработанных выборок.

Результаты:

```text
data/processed/train.csv
data/processed/test.csv
```

### Stage `train`

Выполняет:

- обучение нескольких моделей;
- модельные эксперименты;
- подбор гиперпараметров;
- расчёт метрик;
- построение ROC curve;
- сохранение результатов экспериментов;
- сохранение финальной модели.

Результаты:

```text
models/credit_default_model.pkl
experiment_results.csv
roc_curve.png
metrics.json
```

Полный pipeline:

```bash
dvc repro
```

Если входные данные, код и зависимости стадии не изменились, DVC пропускает уже актуальные стадии.

### Ограничение DVC

DVC remote в проекте не настроен.

Поэтому DVC управляет версиями и зависимостями данных локально, но после свежего `git clone` не может автоматически загрузить исходный dataset или ML-артефакты из удалённого DVC storage.

---

## Артефакты pipeline

| Артефакт | Назначение |
|---|---|
| `data/processed/train.csv` | обучающая выборка |
| `data/processed/test.csv` | тестовая выборка |
| `models/credit_default_model.pkl` | финальная обученная модель |
| `experiment_results.csv` | результаты модельных экспериментов |
| `metrics.json` | метрики финальной модели |
| `roc_curve.png` | ROC curve |
| `dvc.lock` | зафиксированное состояние DVC pipeline |

---

## Тестирование и качество кода

Для unit-тестирования используется **pytest**.

Проверяются:

1. корректность feature engineering;
2. успешная validation корректных данных;
3. отклонение данных с аномальным возрастом.

Запуск:

```bash
python -m pytest -v
```

На момент финальной локальной проверки:

```text
3 passed
```

Для форматирования используется **Black**:

```bash
black --check src tests
```

Для linting используется **Flake8**:

```bash
flake8 src tests
```

После фиксации версий зависимостей проверки были повторно успешно выполнены.

---

## Continuous Integration

CI реализован через **GitHub Actions**.

Workflow находится в:

```text
.github/workflows/ci.yml
```

Он автоматически запускается при:

```text
push
pull_request
```

CI выполняет:

```text
Checkout repository
        ↓
Set up Python 3.12
        ↓
Install dependencies
        ↓
Black
        ↓
Flake8
        ↓
Unit tests
        ↓
Great Expectations validation tests
```

Актуальная версия ветки `main` успешно проходит CI.

---

## FastAPI

Inference-сервис реализован на **FastAPI**.

Основной endpoint:

```text
POST /predict
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

API принимает исходные признаки клиента, выполняет те же feature engineering преобразования, которые использовались при обучении, и передаёт подготовленные данные сохранённой sklearn-модели.

Категориальные значения соответствуют кодировкам исходного dataset. Полная схема входного объекта автоматически отображается в Swagger.

### Пример запроса

```json
{
  "LIMIT_BAL": 200000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 30,
  "PAY_0": 0,
  "PAY_2": 0,
  "PAY_3": 0,
  "PAY_4": 0,
  "PAY_5": 0,
  "PAY_6": 0,
  "BILL_AMT1": 50000,
  "BILL_AMT2": 48000,
  "BILL_AMT3": 47000,
  "BILL_AMT4": 45000,
  "BILL_AMT5": 43000,
  "BILL_AMT6": 42000,
  "PAY_AMT1": 5000,
  "PAY_AMT2": 5000,
  "PAY_AMT3": 5000,
  "PAY_AMT4": 5000,
  "PAY_AMT5": 5000,
  "PAY_AMT6": 5000
}
```

Пример ответа из контрольного запуска:

```json
{
  "default_prediction": 0,
  "default_probability": 0.0530176895
}
```

Где:

- `default_prediction` — предсказанный класс;
- `default_probability` — оценённая вероятность дефолта.

### Два уровня валидации

В проекте используются два механизма validation для разных этапов ML lifecycle.

**Great Expectations** используется в обучающем data pipeline:

```text
dataset
   ↓
Great Expectations
   ↓
feature engineering
   ↓
training
```

Его задача — контролировать набор данных до подготовки и обучения модели.

**Pydantic** используется на уровне inference API:

```text
API request
   ↓
Pydantic
   ↓
feature engineering
   ↓
prediction
```

Его задача — валидировать каждый отдельный запрос.

Например, для возраста задан диапазон:

```text
18 <= AGE <= 100
```

Запрос с:

```text
AGE = 150
```

возвращает:

```text
422 Unprocessable Entity
```

и не передаётся модели.

Таким образом, Pydantic validation не заменяет Great Expectations: эти механизмы защищают разные этапы системы.

---

## Docker

FastAPI-приложение контейнеризировано с помощью **Docker**.

Базовый образ:

```text
python:3.12-slim
```

При сборке устанавливаются зависимости из:

```text
requirements.txt
```

Контейнер запускает:

```text
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Сборка:

```bash
docker build -t credit-scoring-api .
```

Запуск:

```bash
docker run --rm -p 8000:8000 credit-scoring-api
```

На момент финальной проверки:

- Docker image успешно собран;
- FastAPI успешно запущен внутри контейнера;
- Swagger доступен;
- `POST /predict` внутри Docker возвращает `200 OK`.

### Важное ограничение

Текущий `Dockerfile` ожидает наличие:

```text
models/credit_default_model.pkl
```

Поэтому после свежего клонирования рекомендуемый порядок:

```text
добавить исходный CSV
        ↓
dvc repro
        ↓
docker build
        ↓
docker run
```

---

## Data Drift Monitoring

Для демонстрации базового monitoring реализован скрипт:

```text
src/monitoring/drift.py
```

Reference dataset:

```text
data/processed/train.csv
```

Случайная подвыборка из:

```text
data/processed/test.csv
```

используется как имитация incoming data.

Размер incoming sample:

```text
100 наблюдений
```

PSI рассчитывается для:

```text
LIMIT_BAL
AGE
BILL_AMT1
```

### Интерпретация PSI

В демонстрационном скрипте используются пороги:

```text
PSI < 0.10          — значимого drift нет
0.10 <= PSI < 0.25  — умеренный drift
PSI >= 0.25         — значительный drift
```

Пример результата контрольного запуска:

| Feature | PSI | Интерпретация |
|---|---:|---|
| LIMIT_BAL | 0.1009 | умеренный drift |
| AGE | 0.0905 | значимого drift нет |
| BILL_AMT1 | 0.1676 | умеренный drift |

Дополнительно те же 100 наблюдений отправляются в работающий FastAPI endpoint.

Контрольный запуск:

```text
API predictions received: 100
Средняя вероятность дефолта: 0.2298
```

### Запуск мониторинга

Сначала необходимо запустить API, например через Docker:

```bash
docker run --rm -p 8000:8000 credit-scoring-api
```

Затем в другом терминале:

```bash
python -m src.monitoring.drift
```

### Ограничение мониторинга

Этот блок демонстрирует **механизм data drift monitoring**, а не подтверждает наличие реального production drift.

Причины:

- incoming sample взят из того же исходного dataset;
- используется только 100 наблюдений;
- отсутствует реальный production data stream.

Полученные значения PSI показывают работу алгоритма, но не должны интерпретироваться как доказательство изменения поведения реальных клиентов.

---

## Структура проекта

```text
credit-scoring-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_data_overview.ipynb
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── validation.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── pipeline.py
│   │   └── train.py
│   └── monitoring/
│       └── drift.py
├── tests/
│   ├── test_features.py
│   └── test_validation.py
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Воспроизводимость

Версии Python-библиотек зафиксированы в:

```text
requirements.txt
```

После фиксации зависимостей повторно были успешно проверены:

- установка окружения;
- unit-тесты;
- Black;
- Flake8;
- Docker build;
- запуск FastAPI внутри нового Docker image;
- inference через `POST /predict`.

`requirements.txt` является единственным источником актуальных версий библиотек и поэтому не дублируется в README.

---

## Ограничения

Проект является учебным и **не предназначен для непосредственного принятия реальных кредитных решений**.

Основные ограничения:

- DVC remote не настроен;
- MLflow Tracking Store используется локально;
- модель не развёрнута на внешнем production-сервере;
- Docker используется локально;
- monitoring работает на тестовой подвыборке, а не на production data;
- decision threshold отдельно не оптимизировался под бизнес-стоимость ошибок;
- стоимость False Positive и False Negative не моделировалась;
- не реализованы model registry, автоматический retraining и production alerting;
- текущий набор unit-тестов покрывает основные учебные сценарии, но не является полноценным production test suite;
- не проводилась отдельная оценка fairness/bias модели по демографическим группам.

Наличие демографических признаков в исходных данных требует отдельного анализа fairness до любого реального использования модели в кредитном процессе.

---

## Технологии

- Python 3.12
- NumPy
- pandas
- scikit-learn
- matplotlib
- Great Expectations
- MLflow
- DVC
- pytest
- Black
- Flake8
- Git
- GitHub Actions
- FastAPI
- Pydantic
- Uvicorn
- Docker

---

## Итог

Проект демонстрирует end-to-end учебный MLOps workflow для задачи кредитного скоринга: от валидации и подготовки данных до experiment tracking, воспроизводимого pipeline, CI, inference API, контейнеризации и демонстрационного мониторинга data drift.