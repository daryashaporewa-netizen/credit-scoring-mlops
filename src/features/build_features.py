def build_features(df):
    """Создаёт дополнительные признаки на основе истории клиента."""

    df = df.copy()

    bill_columns = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]

    payment_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    delay_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    # Средняя доля использованного кредитного лимита
    df["AVG_UTILIZATION"] = df[bill_columns].mean(axis=1) / df["LIMIT_BAL"]

    # Средний размер платежа относительно кредитного лимита
    df["AVG_PAYMENT_RATIO"] = df[payment_columns].mean(axis=1) / df["LIMIT_BAL"]

    # Максимальная просрочка за последние шесть месяцев
    df["MAX_DELAY"] = df[delay_columns].max(axis=1)

    # Количество месяцев с просрочкой
    df["DELAY_MONTHS"] = (df[delay_columns] > 0).sum(axis=1)

    return df
