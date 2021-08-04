from gym_anytrading.envs import StocksEnv


def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]
    return prices, signal_features


class Environment(StocksEnv):
    _process_data = my_process_data

    def __init__(self, df, window_size, frame_bound):
        super().__init__(df, window_size, frame_bound)
