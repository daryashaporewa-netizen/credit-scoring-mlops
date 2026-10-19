# Импортируем необходимые библиотеки

from pathlib import Path
import great_expectations as gx
import pandas as pd

# Ожидаемые столбцы датасета

required_columns = [
    "ID",
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
    "default.payment.next.month",
]

# Ожидаемые типы данных

expected_types = {
    "ID": "int64",
    "LIMIT_BAL": "float64",
    "SEX": "int64",
    "EDUCATION": "int64",
    "MARRIAGE": "int64",
    "AGE": "int64",
    "PAY_0": "int64",
    "PAY_2": "int64",
    "PAY_3": "int64",
    "PAY_4": "int64",
    "PAY_5": "int64",
    "PAY_6": "int64",
    "BILL_AMT1": "float64",
    "BILL_AMT2": "float64",
    "BILL_AMT3": "float64",
    "BILL_AMT4": "float64",
    "BILL_AMT5": "float64",
    "BILL_AMT6": "float64",
    "PAY_AMT1": "float64",
    "PAY_AMT2": "float64",
    "PAY_AMT3": "float64",
    "PAY_AMT4": "float64",
    "PAY_AMT5": "float64",
    "PAY_AMT6": "float64",
    "default.payment.next.month": "int64",
}


def validate_data(df):
    """Проверяет DataFrame по заданным правилам качества данных."""

    # Создаём временный контекст Great Expectations
    context = gx.get_context(mode="ephemeral")

    # Подключаем DataFrame к Great Expectations
    data_source = context.data_sources.add_pandas("credit_data")
    data_asset = data_source.add_dataframe_asset(
        name="credit_card_data"
    )
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "full_dataset"
    )

    # Создаём единый набор правил
    suite = gx.ExpectationSuite(name="credit_data_suite")
    suite = context.suites.add(suite)

    # Проверяем структуру датасета
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=required_columns,
            exact_match=True,
        )
    )

    # Проверяем отсутствие пропусков
    for column in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column
            )
        )

    # Проверяем типы данных
    for column, expected_type in expected_types.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnTypeToBe(
                column=column,
                type_=expected_type,
            )
        )

    # Проверяем допустимый диапазон возраста
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="AGE",
            min_value=18,
            max_value=100,
        )
    )

    # Проверяем допустимые значения пола
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="SEX",
            value_set=[1, 2],
        )
    )

    # Проверяем значения целевой переменной
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="default.payment.next.month",
            value_set=[0, 1],
        )
    )

    suite.save()

    # Связываем набор правил с данными
    validation_definition = gx.ValidationDefinition(
        name="credit_data_validation",
        data=batch_definition,
        suite=suite,
    )

    validation_definition = context.validation_definitions.add(
        validation_definition
    )

    # Запускаем все проверки одной общей валидацией
    validation_result = validation_definition.run(
        batch_parameters={"dataframe": df}
    )

    return validation_result.success

if __name__ == "__main__":
    # Загружаем исходный датасет

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"

    df = pd.read_csv(data_path)

    result = validate_data(df)

    print("Итоговая валидация:", result)