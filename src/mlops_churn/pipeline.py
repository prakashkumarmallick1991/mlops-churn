from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.mlops_churn.data import create_dataset
from src.mlops_churn.evaluate import evaluate_model
from src.mlops_churn.validation import validate_data




MODEL_PATH = Path("models/churn_model.pkl")


def run_training():

    # 1. Load data
    df = create_dataset()

    
    df = validate_data(df)
    print("Data validation passed")

    # 2. Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # 6. Save model
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