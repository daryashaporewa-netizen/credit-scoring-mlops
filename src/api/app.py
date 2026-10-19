from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.build_features import build_features

# Пути проекта и модели
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "credit_default_model.pkl"

# Загружаем обученную модель
model = joblib.load(model_path)


# Создаём FastAPI-приложение
app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта клиента.",
    version="1.0.0",
)


class ClientData(BaseModel):
    """Входные данные клиента."""

    LIMIT_BAL: float = Field(gt=0)

    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(ge=18, le=100)

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Ответ модели."""

    default_prediction: int
    default_probability: float


@app.get("/")
def root():
    """Проверка доступности API."""

    return {"message": "Credit Default Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    """Возвращает класс и вероятность дефолта клиента."""

    # Превращаем входной JSON в DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Создаём те же признаки, которые использовались при обучении
    input_df = build_features(input_df)

    # Сохраняем тот же набор и порядок признаков,
    # который использовался при обучении модели
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[list(model.feature_names_in_)]

    # Получаем прогноз
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return PredictionResponse(
        default_prediction=int(prediction),
        default_probability=float(probability),
    )
