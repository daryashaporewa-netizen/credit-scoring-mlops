import pandas as pd

from src.data.validation import validate_data


def make_valid_data():
    """Создаёт небольшой корректный датасет для тестирования валидации."""

    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [20000.0, 50000.0],
            "SEX": [1, 2],
            "EDUCATION": [2, 1],
            "MARRIAGE": [1, 2],
            "AGE": [24, 35],
            "PAY_0": [0, 1],
            "PAY_2": [0, 0],
            "PAY_3": [0, 0],
            "PAY_4": [0, 0],
            "PAY_5": [0, 0],
            "PAY_6": [0, 0],
            "BILL_AMT1": [3913.0, 12000.0],
            "BILL_AMT2": [3102.0, 11000.0],
            "BILL_AMT3": [689.0, 10000.0],
            "BILL_AMT4": [0.0, 9000.0],
            "BILL_AMT5": [0.0, 8000.0],
            "BILL_AMT6": [0.0, 7000.0],
            "PAY_AMT1": [0.0, 1000.0],
            "PAY_AMT2": [689.0, 1000.0],
            "PAY_AMT3": [0.0, 1000.0],
            "PAY_AMT4": [0.0, 1000.0],
            "PAY_AMT5": [0.0, 1000.0],
            "PAY_AMT6": [0.0, 1000.0],
            "default.payment.next.month": [1, 0],
        }
    )

    float_columns = [
        "LIMIT_BAL",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    int_columns = [
        "ID",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "default.payment.next.month",
    ]

    df[float_columns] = df[float_columns].astype("float64")
    df[int_columns] = df[int_columns].astype("int64")

    return df


def test_data_validation():
    """Корректные данные должны проходить валидацию."""

    df = make_valid_data()

    assert validate_data(df)


def test_invalid_age():
    """Аномальный возраст должен провалить валидацию."""

    df = make_valid_data()
    df.loc[0, "AGE"] = 150

    assert not validate_data(df)
