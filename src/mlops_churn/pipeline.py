from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.mlops_churn.data import create_dataset
from src.mlops_churn.evaluate import evaluate_model
from src.mlops_churn.validation import validate_data




MODEL_PATH = Path("models/churn_model.pkl")


def run_training():

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():

        # 1. Load data
        df = create_dataset()

        # 2. Validate data
        df = validate_data(df)
        print("Data validation passed")

        # 3. Separate features and target
        X = df.drop(columns=["churn"])
        y = df["churn"]

        # 4. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # 5. Model parameters
        n_estimators = 200
        random_state = 42

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

        # 6. Train
        model.fit(X_train, y_train)

        # 7. Evaluate
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # 8. Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # 9. Log metrics
        mlflow.log_metrics(metrics)

        # 10. Log model
        mlflow.sklearn.log_model(
            model,
            name="churn_model",
        )

        # 11. Save local copy
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        return metrics


if __name__ == "__main__":
    metrics = run_training()

    print("\nTraining complete")
    print("-----------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")