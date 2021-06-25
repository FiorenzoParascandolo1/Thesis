from src.data_utils.preprocessing_utils import standardize_dataframe_cols, download_data, add_period_return, \
    normalize_dataframe_cols

if __name__ == "__main__":
    dataframe = download_data(ticker="TSLA", period="5min")

