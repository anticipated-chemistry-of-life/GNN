import category_encoders as ce
import pandas as pd

def binary_encode_df(data: pd.DataFrame)-> pd.DataFrame:
    encoder = ce.BinaryEncoder(cols=[col for col in data.columns])

    return encoder.fit_transform(data)