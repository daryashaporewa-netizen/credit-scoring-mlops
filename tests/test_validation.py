from pathlib import Path
import pandas as pd
from src.data.validation import validate_data

project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"


def test_data_validation():
    """Проверяем, что исходный датасет проходит валидацию."""

    df = pd.read_csv(data_path)

    assert validate_data(df)


def test_invalid_age():
    """Проверяем, что аномальный возраст не проходит валидацию."""

    df = pd.read_csv(data_path)
    df.loc[0, "AGE"] = 150

    assert not validate_data(df)
