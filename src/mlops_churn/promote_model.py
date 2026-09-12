import mlflow
from mlflow import MlflowClient


MODEL_NAME = "churn_model"


def promote_latest_model():
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        latest_version.version,
    )

    print(
        f"Promoted version {latest_version.version} "
        f"to 'champion'"
    )


if __name__ == "__main__":
    promote_latest_model()