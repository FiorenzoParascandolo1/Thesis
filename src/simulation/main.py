from src.policy.policy import PPO
from src.simulation.environment import Environment
from src.simulation.training import training_loop
import yfinance as yf
import requests
import pandas as pd
import time

params = {
    'Ticker': ["CCL"],
    'Period': "10y",
    'Interval': "1d",
    'EnvType': "stocks-v0",
    'WindowSize': 8102,
    'Lr': 1e-4,
    'Periods': [1, 12, 54, 270],
    'Pixels': 30,
    'EpsClip': 0.1,
    'Gamma': 0.99,
    'LenMemory': 451,
    'Horizon': 45,
    'UpdateTimestamp': 10,
    'Wallet': 100000}


def main():
    """
    import csv
    import requests

    dfs = []

    for i in reversed(range(1, 3)):
        for j in reversed(range(1, 13)):
            with requests.Session() as s:
                CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&' \
                          f'interval=5min&slice=year{i}month{j}&apikey=LM33S122HKMP08J2'
                download = s.get(CSV_URL)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                my_list = list(cr)
                dfs.append(pd.DataFrame(reversed(my_list[1:]), columns=my_list[0]).reset_index())
                print(pd.DataFrame(reversed(my_list[1:]), columns=my_list[0]).head())
                time.sleep(30)

    df = pd.concat(dfs)
    df.to_csv("IBM.csv")
    df_1 = pd.read_csv("IBM.csv")
    df_2 = pd.read_csv("wallet.csv")
    pd.DataFrame({'Close': list(df_1['close'].iloc[(params['WindowSize'] - 1):(df_1.shape[0])]),
                  'Wallet': df_2['0'].iloc[:]}).to_csv("ibm_close_wallet.csv")


    df = yf.download(tickers=params['Ticker'],
                     period=params['Period'],
                     interval=params['Interval'],
                     group_by='ticker',
                     auto_adjust=True,
                     prepost=True,
                     threads=True,
                     proxy=None)
    """

    df = pd.read_csv("IBM.csv")
    env = Environment(df=df.iloc[:8200],
                      window_size=params['WindowSize'],
                      frame_bound=(params['WindowSize'], len(df)),
                      starting_wallet=params['Wallet'])
    policy = PPO(params)
    training_loop(env=env, policy=policy)
    """
    df_1 = pd.read_csv("IBM.csv")
    df_2 = pd.read_csv("wallet.csv")
    print(len(list(df_1['close'].iloc[(params['WindowSize']):(df_1.shape[0])])))
    print(len(df_2['0'].iloc[:]))
    pd.DataFrame({'Close': list(df_1['close'].iloc[(params['WindowSize']):(df_1.shape[0] - 1)]),
                  'Wallet': df_2['0'].iloc[:]}).to_csv("ibm_close_wallet.csv")
    """


if __name__ == "__main__":
    main()
