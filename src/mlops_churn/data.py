import pandas as pd
from sklearn.datasets import make_classification


def create_dataset(n_samples=1000, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        weights=[0.7, 0.3],
        random_state=random_state,
    )

    df = pd.DataFrame(
        X,
        columns=[
            "monthly_charges",
            "tenure",
            "support_calls",
            "login_frequency",
            "contract_length",
        ],
    )

    df["churn"] = y

    return df