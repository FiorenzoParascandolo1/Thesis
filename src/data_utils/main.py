import pandas as pd

from src.data_utils.preprocessing_utils import add_features_on_time

if __name__ == "__main__":
    dataframe = pd.read_csv(
        "s3://aivo-rnd-trading/data/frd-sp500/5min/TSLA_5min.txt",
        header=None,
        names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
        parse_dates=["DateTime"]
    )
    dataframe = add_features_on_time(dataframe)
    print(dataframe.head())
