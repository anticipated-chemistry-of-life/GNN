import category_encoders as ce
import pandas as pd


def binary_encode_df(data: pd.DataFrame) -> pd.DataFrame:
    encoder = ce.BinaryEncoder(cols=[col for col in data.columns])

    return encoder.fit_transform(data)


def hash_encoding(data: pd.DataFrame, n_features=16) -> pd.DataFrame:
    encoder = ce.HashingEncoder(
        cols=[col for col in data.columns], n_components=n_features
    )

    return encoder.fit_transform(data)
