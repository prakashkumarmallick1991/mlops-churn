import mlflow
import os
from mlflow import MlflowClient


MODEL_NAME = "churn_model"

mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

from mlflow import MlflowClient

client = MlflowClient(tracking_uri=mlflow_tracking_uri)

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