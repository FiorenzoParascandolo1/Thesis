from src.simulation.environment import Environment
from src.simulation.training import training_loop
import yfinance as yf

params = {
    'Tickers': "AAPL",
    'Period': "10y",
    'Interval': "1d",
    'EnvType': "stocks-v0",
    'WindowSize': 10,
}


def main():
    df = yf.download(tickers=params['Tickers'],
                     period=params['Period'],
                     interval=params['Interval'],
                     group_by='ticker',
                     auto_adjust=True,
                     prepost=True,
                     threads=True,
                     proxy=None)
    env = Environment(df=df, window_size=params['WindowSize'], frame_bound=(params['WindowSize'], len(df)))
    training_loop(env)


if __name__ == "__main__":
    main()
