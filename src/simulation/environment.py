import gym
import gym_anytrading
import pyfinancialdata
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt


class Environment(object):
    def __init__(self,
                 params: dict):
        """
        :param params: the parameters required to initialize the environment
        """
        df = pyfinancialdata.get_multi_year(provider=params['Provider'],
                                            instrument=params['Instrument'],
                                            years=params['Years'],
                                            time_group=params['TimeGroup'])
        df = df.reset_index()
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Price']
        self.env = gym.make(params['EnvType'],
                            df=df,
                            window_size=params['WindowSize'],
                            frame_bound=params['FrameBound'])

    def reset(self):
        self.env.reset()

    def print_information(self):
        print("custom_env information:")
        print("> shape:", self.env.shape)
        print("> df.shape:", self.env.df.shape)
        print("> prices.shape:", self.env.prices.shape)
        print("> signal_features.shape:", self.env.signal_features.shape)
        print("> max_possible_profit:", self.env.max_possible_profit())

    def render(self):
        self.env.render()
