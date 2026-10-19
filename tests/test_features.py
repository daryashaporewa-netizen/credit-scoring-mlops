import pandas as pd

from src.features.build_features import build_features


def test_build_features():
    """Проверяем создание новых признаков."""

    df = pd.DataFrame(
        {
            "LIMIT_BAL": [100_000],
            "BILL_AMT1": [50_000],
            "BILL_AMT2": [50_000],
            "BILL_AMT3": [50_000],
            "BILL_AMT4": [50_000],
            "BILL_AMT5": [50_000],
            "BILL_AMT6": [50_000],
            "PAY_AMT1": [10_000],
            "PAY_AMT2": [10_000],
            "PAY_AMT3": [10_000],
            "PAY_AMT4": [10_000],
            "PAY_AMT5": [10_000],
            "PAY_AMT6": [10_000],
            "PAY_0": [2],
            "PAY_2": [0],
            "PAY_3": [1],
            "PAY_4": [0],
            "PAY_5": [-1],
            "PAY_6": [0],
        }
    )

    result = build_features(df)

    assert result["AVG_UTILIZATION"].iloc[0] == 0.5
    assert result["AVG_PAYMENT_RATIO"].iloc[0] == 0.1
    assert result["MAX_DELAY"].iloc[0] == 2
    assert result["DELAY_MONTHS"].iloc[0] == 2
