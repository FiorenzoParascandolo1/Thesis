import pandas as pd


def add_features_on_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add time information as columns:
    - WeekDay: [0, 0.2, 0.4, 0.6, 0.8, 1] <-> [Monday, Tuesday, Wednesday, Thursday, Friday]
    - MonthDay: [1 / 31, ..., 1]
    - Month: [1/12, ..., 1]
    - Hour: [0, 1/23, ..., 1]
    - Minute: [0, 1/59, ..., 1]
    - TradeHour: 1 if the market is open in that time else 0

    :param dataframe: dataframe to be processed.
    :return: processed dataframe.
    """

    MAX_WEEK_DAY = 4
    MAX_MONTH_DAY = 31
    MAX_MONTH = 12
    MAX_HOUR = 23
    MAX_MINUTE = 59
    STARTING_TRADE_HOUR = 9 / 23
    FINAL_TRADE_HOUR = 16 / 23

    dataframe['WeekDay'] = dataframe['DateTime'].apply(lambda x: x.weekday() / MAX_WEEK_DAY)
    dataframe['MonthDay'] = dataframe['DateTime'].apply(lambda x: x.day / MAX_MONTH_DAY)
    dataframe['Month'] = dataframe['DateTime'].apply(lambda x: x.month / MAX_MONTH)
    dataframe['Hour'] = dataframe['DateTime'].apply(lambda x: x.hour / MAX_HOUR)
    dataframe['Minute'] = dataframe['DateTime'].apply(lambda x: x.minute / MAX_MINUTE)
    dataframe['TradeHour'] = 1 * (dataframe['Hour'] > STARTING_TRADE_HOUR) * (dataframe['Hour'] < FINAL_TRADE_HOUR)

    return dataframe

