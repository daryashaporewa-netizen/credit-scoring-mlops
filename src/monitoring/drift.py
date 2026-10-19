import json
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

API_URL = "http://127.0.0.1:8000/predict"

TARGET_COLUMN = "default.payment.next.month"

API_COLUMNS = [
    "LIMIT_BAL",
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

MONITORED_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
]


def calculate_psi(expected, actual, bins=10):
    """Вычисляет Population Stability Index."""

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    boundaries = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))

    if len(boundaries) < 2:
        return 0.0

    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=boundaries)
    actual_counts, _ = np.histogram(actual, bins=boundaries)

    expected_percent = expected_counts / len(expected)
    actual_percent = actual_counts / len(actual)

    epsilon = 1e-6

    expected_percent = np.clip(expected_percent, epsilon, None)
    actual_percent = np.clip(actual_percent, epsilon, None)

    psi = np.sum(
        (actual_percent - expected_percent) * np.log(actual_percent / expected_percent)
    )

    return float(psi)


def interpret_psi(psi):
    """Интерпретирует значение PSI."""

    if psi < 0.1:
        return "нет значимого drift"
    if psi < 0.25:
        return "умеренный drift"

    return "значительный drift"


def send_to_api(row):
    """Отправляет одного клиента в FastAPI и возвращает вероятность дефолта."""

    payload = {
        column: row[column].item() if hasattr(row[column], "item") else row[column]
        for column in API_COLUMNS
    }

    data = json.dumps(payload).encode("utf-8")

    api_request = request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(api_request, timeout=10) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["default_probability"]


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    incoming_data = test_df.sample(
        n=min(100, len(test_df)),
        random_state=42,
    ).copy()

    print("=== DATA DRIFT MONITORING ===")
    print(f"Reference observations: {len(train_df)}")
    print(f"Incoming observations: {len(incoming_data)}")
    print()

    print("PSI по входным признакам:")

    for feature in MONITORED_FEATURES:
        psi = calculate_psi(
            train_df[feature],
            incoming_data[feature],
        )

        print(f"{feature}: PSI = {psi:.4f} " f"({interpret_psi(psi)})")

    print()
    print("Отправка новых наблюдений в API...")

    probabilities = []

    for _, row in incoming_data.iterrows():
        probability = send_to_api(row)
        probabilities.append(probability)

    print(f"API predictions received: {len(probabilities)}")
    print("Средняя вероятность дефолта: " f"{np.mean(probabilities):.4f}")

    print()
    print("Мониторинг завершён успешно.")


if __name__ == "__main__":
    main()
