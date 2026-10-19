from pathlib import Path
import json
import joblib

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.models.pipeline import create_pipeline


project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed"

target = "default.payment.next.month"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit_Default_Prediction")


categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def calculate_metrics(model, X_test, y_test):
    """Рассчитывает основные метрики классификации."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics, y_prob


def run_experiment(
    name,
    classifier,
    numeric_features,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Обучает модель и записывает эксперимент в MLflow."""

    model = create_pipeline(
        numeric_features,
        categorical_features,
    )

    # Заменяем classifier внутри pipeline
    model.set_params(classifier=classifier)

    model.fit(X_train, y_train)

    metrics, y_prob = calculate_metrics(
        model,
        X_test,
        y_test,
    )

    with mlflow.start_run(run_name=name):

        mlflow.log_param(
            "model_type",
            classifier.__class__.__name__,
        )

        mlflow.log_params(classifier.get_params())
        mlflow.log_metrics(metrics)

        # ROC-кривая
        roc_path = project_root / "roc_curve.png"

        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title(f"ROC-кривая — {name}")
        plt.savefig(roc_path)
        plt.close()

        mlflow.log_artifact(str(roc_path))

        # Сохраняем модель в MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

    print(
        f"{name}: "
        f"ROC-AUC={metrics['roc_auc']:.3f}, "
        f"Precision={metrics['precision']:.3f}, "
        f"Recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )

    return model, metrics


def main():
    # Загружаем подготовленные данные
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    X_train = train.drop(columns=target)
    y_train = train[target]

    X_test = test.drop(columns=target)
    y_test = test[target]

    numeric_features = [
        column
        for column in X_train.columns
        if column not in categorical_features
    ]

    # GridSearchCV для Gradient Boosting

    pipeline = create_pipeline(
        numeric_features,
        categorical_features,
    )

    search = GridSearchCV(
        pipeline,
        {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print("Лучшие параметры GridSearchCV:")
    print(search.best_params_)

    best_params = {
        key.replace("classifier__", ""): value
        for key, value in search.best_params_.items()
    }

    # Пять MLflow-экспериментов

    experiments = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
        (
            "logistic_regression_balanced",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "gradient_boosting_default",
            GradientBoostingClassifier(
                random_state=42,
            ),
        ),
        (
            "gradient_boosting_tuned",
            GradientBoostingClassifier(
                **best_params,
                random_state=42,
            ),
        ),
        (
            "gradient_boosting_alternative",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                random_state=42,
            ),
        ),
    ]

    results = {}

    for name, classifier in experiments:
        model, metrics = run_experiment(
            name,
            classifier,
            numeric_features,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results[name] = {
            "model": model,
            "metrics": metrics,
        }

    # Сохраняем результаты всех экспериментов
    results_table = pd.DataFrame([
        {
            "model": name,
            **result["metrics"],
        }
        for name, result in results.items()
    ])

    results_table.to_csv(
        project_root / "experiment_results.csv",
        index=False,
    )

    print("\nРезультаты экспериментов:")
    print(results_table.round(3))

    # Сохраняем лучшую модель и метрики в файлы
    final_model = results["gradient_boosting_tuned"]["model"]
    final_metrics = results["gradient_boosting_tuned"]["metrics"]

    joblib.dump(
        final_model,
        project_root / "models" / "credit_default_model.pkl",
    )

    with open(project_root / "metrics.json", "w") as file:
        json.dump(final_metrics, file, indent=4)

    # ROC-кривая итоговой модели
    _, final_prob = calculate_metrics(
        final_model,
        X_test,
        y_test,
    )

    RocCurveDisplay.from_predictions(y_test, final_prob)
    plt.title("ROC-кривая — финальная модель")
    plt.savefig(project_root / "roc_curve.png")
    plt.close()


if __name__ == "__main__":
    main()