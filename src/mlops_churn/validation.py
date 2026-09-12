import pandera.pandas as pa


class ChurnSchema(pa.DataFrameModel):
    monthly_charges: float
    tenure: float
    support_calls: float
    login_frequency: float
    contract_length: float

    churn: int = pa.Field(isin=[0, 1])

    class Config:
        strict = True


def validate_data(df):
    return ChurnSchema.validate(df)