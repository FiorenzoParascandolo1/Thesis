import pandas as pd
import numpy as np

MAX_WEEK_DAY = 4
MAX_MONTH_DAY = 31
MAX_MONTH = 12
MAX_HOUR = 23
MAX_MINUTE = 55
STARTING_TRADE_HOUR = 9 / 23
FINAL_TRADE_HOUR = 16 / 23


def add_features_on_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add time information as columns:
    - WeekDay: [0, 0.2, 0.4, 0.6, 0.8, 1] <-> [Monday, Tuesday, Wednesday, Thursday, Friday]
    - MonthDay: [1 / 31, ..., 1]
    - Month: [1/12, ..., 1]
    - Hour: [0, 1/23, ..., 1]
    - Minute: [0, 5/55 ..., 1]
    - TradeHour: 1 if the market is open in that time else 0

    :param dataframe: dataframe to be processed.
    :return: processed dataframe.
    """

    dataframe['WeekDay'] = dataframe['DateTime'].apply(lambda x: x.weekday() / MAX_WEEK_DAY)
    dataframe['MonthDay'] = dataframe['DateTime'].apply(lambda x: x.day / MAX_MONTH_DAY)
    dataframe['Month'] = dataframe['DateTime'].apply(lambda x: x.month / MAX_MONTH)
    dataframe['Hour'] = dataframe['DateTime'].apply(lambda x: x.hour / MAX_HOUR)
    dataframe['Minute'] = dataframe['DateTime'].apply(lambda x: x.minute / MAX_MINUTE)
    dataframe['TradeHour'] = 1 * (dataframe['Hour'] > STARTING_TRADE_HOUR) * (dataframe['Hour'] < FINAL_TRADE_HOUR)

    return dataframe


def add_period_return(dataframe: pd.DataFrame,
                      period: int = 1,
                      method: str = "linear") -> pd.DataFrame:
    """
    Add the column 'Return' to the input dataframe. It represents.
    Most financial studies involve returns, instead of prices, of assets.
    There are two main advantages of using returns.
    First, for average investors, return of an asset is a complete and scale-free summary of the investment opportunity.
    Second, return series are easier to handle than price series because the former have more attractive statistical
    properties.

    :param dataframe: dataframe to be processed.
    :param period: Period return
    :param method: "linear" or "log"
    :return: processed dataframe.
    """

    if 'Return' not in dataframe.columns:
        dataframe['Return'] = 1
    # Base case
    if period == 0:
        if method == "linear":
            dataframe['Return'] = dataframe['Return'] - 1
        else:
            dataframe['Return'] = np.log(dataframe['Return'])
        return dataframe
    dataframe['Return'] = \
        dataframe['Return'] * dataframe['Close'].shift(periods=period - 1) / dataframe['Close'].shift(periods=period)
    # Recursive call
    return add_period_return(dataframe=dataframe, period=period - 1)


def standardize_dataframe_cols(dataframe: pd.DataFrame,
                               col_names: list = None) -> pd.DataFrame:
    """
    Standardize the specified columns of the input dataframe.

    :param dataframe: dataframe to standardize.
    :param col_names: columns to standardize.
    :return: the standardized dataframe.
    """
    if col_names is None:
        col_names = ["Open", "High", "Low", "Close", "Volume"]

    dataframe[col_names] = (dataframe[col_names] - dataframe[col_names].mean()) / dataframe[col_names].std()

    return dataframe
