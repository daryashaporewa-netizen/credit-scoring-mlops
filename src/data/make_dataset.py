from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.validation import validate_data
from src.features.build_features import build_features


project_root = Path(__file__).resolve().parents[2]

raw_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"
processed_path = project_root / "data" / "processed"


def main():
    # Загружаем и проверяем исходные данные
    df = pd.read_csv(raw_path)

    if not validate_data(df):
        raise ValueError("Данные не прошли валидацию")

    # Создаём новые признаки и удаляем ID
    df = build_features(df)
    df = df.drop(columns="ID")

    # Делим данные на train и test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["default.payment.next.month"],
    )

    # Сохраняем результат
    train_df.to_csv(processed_path / "train.csv", index=False)
    test_df.to_csv(processed_path / "test.csv", index=False)

    print("Данные подготовлены успешно")
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)


if __name__ == "__main__":
    main()