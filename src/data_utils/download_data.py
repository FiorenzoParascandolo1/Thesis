import csv
import requests
import pandas as pd
import time


def download_data(symbol: str) -> None:
    dfs = []

    for i in reversed(range(1, 2)):
        for j in reversed(range(1, 13)):
            with requests.Session() as s:
                print(i, j)
                CSV_URL = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=5min&' \
                          f'slice=year{i}month{j}&outputsize=full&apikey=0DPQQ5IM2U8DZCCV&'
                """
                CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=CCL&' \
                          f'interval=5min&slice=year{i}month{j}&apikey=LM33S122HKMP08J2'
                """
                download = s.get(CSV_URL)
                data = download.json()
                df = pd.DataFrame.from_dict(data['Time Series FX (5min)'], orient='index')
                print(df.iloc[::-1])
                """
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                my_list = list(cr)
                print(my_list)
                dfs.append(pd.DataFrame(reversed(my_list[1:]), columns=my_list[0]).reset_index())
                print(pd.DataFrame(reversed(my_list[1:]), columns=my_list[0]).head())
                """
                time.sleep(15)

    df = pd.concat(dfs)
    df.to_csv(symbol + '.csv')

download_data('EURUSD')
