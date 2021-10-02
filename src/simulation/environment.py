from gym_anytrading.envs import StocksEnv, Actions, Positions

from src.wallet.wallet import Wallet


def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['open', 'high', 'low', 'volume', 'close']].to_numpy()[start:end]
    return prices, signal_features


def compute_kelly(win_percent):
    return win_percent - (1 - win_percent) / (win_percent / (1 - win_percent))


class Environment(StocksEnv):
    _process_data = my_process_data

    def __init__(self, df, window_size, frame_bound, starting_wallet):
        super().__init__(df, window_size, frame_bound)
        self.trade_fee_bid_percent = 0.0  # unit
        self.trade_fee_ask_percent = 0.0  # unit
        self._start_tick = self.window_size
        self._end_tick = len(self.prices)
        self.wallet = Wallet(starting_wallet, self.prices[self._start_tick - 1])

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick
        self._position = Positions.Short
        obs = self._get_observation()

        return obs

    def _calculate_reward(self, action):

        price_1 = price_2 = denominator = 0

        if action[0] == Actions.Sell.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._last_trade_tick]
            denominator = price_2
        if action[0] == Actions.Buy.value and self._position == Positions.Short:
            price_1 = self.prices[self._last_trade_tick]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_1
        if action[0] == Actions.Buy.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 1]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_2
        if action[0] == Actions.Sell.value and self._position == Positions.Short:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._current_tick - 1]
            denominator = price_1

        step_reward = (price_1 - price_2) / denominator * self.wallet.cap_inv

        return step_reward

    def step(self, action):
        self._done = False

        step_reward = self._calculate_reward(action)
        info = self.wallet.step(action, self.prices[self._last_trade_tick], self.prices[self._current_tick - 2],
                                self._position, step_reward)
        if action[0] == Actions.Buy.value and self._position == Positions.Short or \
                action[0] == Actions.Sell.value and self._position == Positions.Long:
            self._last_trade_tick = self._current_tick - 2

        if ((action[0] == Actions.Buy.value and self._position == Positions.Short) or
                (action[0] == Actions.Sell.value and self._position == Positions.Long)):
            self._position = self._position.opposite()

        self._current_tick += 1
        observation = self._get_observation()

        if self._current_tick == self._end_tick:
            self._done = True

        return observation, step_reward, self._done, info

    def get_position(self):
        return self._position

    def render_performances(self):
        self.wallet.render_all(self.prices[(self.window_size - 1):-1])
