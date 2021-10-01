from gym_anytrading.envs import StocksEnv, Actions, Positions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['time', 'open', 'high', 'low', 'volume', 'close']].to_numpy()[start:end]
    return prices, signal_features


def compute_kelly(win_percent):
    return win_percent - (1 - win_percent) / (win_percent / (1 - win_percent))


def max_dd(wallet_series):
    cumulative_returns = pd.Series(wallet_series)
    highwatermarks = cumulative_returns.cummax()
    drawdowns = 1 - (1 + cumulative_returns) / (1 + highwatermarks)
    max_drawdown = max(drawdowns)

    return max_drawdown * 100


class Environment(StocksEnv):
    _process_data = my_process_data

    def __init__(self, df, window_size, frame_bound, wallet):
        super().__init__(df, window_size, frame_bound)
        self.trade_fee_bid_percent = 0.0  # unit
        self.trade_fee_ask_percent = 0.0  # unit
        self._start_tick = self.window_size
        self.starting_wallet = wallet
        self.equity_benchmark = [0]
        self.equity_trading_system = [0]
        self.wallet = self.starting_wallet
        self.wallet_series = [self.starting_wallet]
        self.pl_series = [0]
        self.cap_inv = 0.5 * self.starting_wallet
        self._end_tick = len(self.prices)
        self.tot_operation = 0
        self.profit_trades = 0
        self.total_gain = 0
        self.total_loss = 0

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick
        self._position = Positions.Short
        self._position_history = [0]
        self._total_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        obs = self._get_observation()
        return obs

    def _calculate_reward(self, action):

        price_1 = price_2 = denominator = 0
        done = False

        if action[0] == Actions.Sell.value and self._position == Positions.Long:
            update_profit = True
        else:
            update_profit = False
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

        if (action[0] == Actions.Sell.value and self._position == Positions.Long) or \
                (action[0] == Actions.Buy.value and self._position == Positions.Long):
            self.equity_trading_system.append((self._total_reward + (self.prices[self._last_trade_tick] - price_2)
                                               / self.prices[self._last_trade_tick] * self.cap_inv)
                                              / self.starting_wallet)
            self.pl_series.append((self.prices[self._last_trade_tick] - price_2)
                                  / self.prices[self._last_trade_tick] * self.cap_inv)
            if self.equity_trading_system[-1] > 1:
                self.equity_trading_system[-1] -= 1
        else:
            self.equity_trading_system.append(self.equity_trading_system[-1])
            self.pl_series.append(0)
        self.equity_benchmark.append((self.prices[self._current_tick - 2] -
                                      self.prices[self._start_tick - 1]) / self.prices[self._start_tick - 1])
        if self.equity_benchmark[-1] > 1:
            self.equity_benchmark[-1] -= 1

        step_reward = (price_1 - price_2) / denominator * self.cap_inv

        if ((action[0] == Actions.Sell.value and self._position == Positions.Long)
                or (action[0] == Actions.Buy.value and self._position == Positions.Short)):
            self.cap_inv = action[1][0][action[0]].item() * self.wallet
            done = True

        return step_reward, update_profit, done

    def step(self, action):
        self._done = False
        save_position = False
        step_reward, update_profit, done = self._calculate_reward(action)
        if action[0] == Actions.Buy.value and self._position == Positions.Short or \
                action[0] == Actions.Sell.value and self._position == Positions.Long:
            self._last_trade_tick = self._current_tick - 2
        self._current_tick += 1
        if update_profit:
            self._total_reward += step_reward
            self._update_profit(step_reward)
        self.wallet_series.append(self.wallet)

        if ((action[0] == Actions.Buy.value and self._position == Positions.Short) or
                (action[0] == Actions.Sell.value and self._position == Positions.Long)):
            self._position = self._position.opposite()
            save_position = True
        if save_position:
            position_value = self._position.value
        else:
            position_value = None
        self._position_history.append(position_value)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            done=done)
        self._update_history(info)

        if self._current_tick == self._end_tick:
            self._done = True

        return observation, step_reward, self._done, info

    def _update_profit(self, step_reward):
        self.wallet += step_reward
        self._total_profit = (self._total_reward / self.starting_wallet) * 100
        self.tot_operation += 1
        if step_reward > 0:
            self.profit_trades += 1
            self.total_gain += step_reward
        else:
            self.total_loss += step_reward

    def get_position(self):
        return self._position

    def render_all(self, mode='human'):
        fig, (ax1, ax2) = plt.subplots(2)

        fig.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Percent Profit: %.6f" % self._total_profit + ' ~ ' +
            "Number of trades: %d" % self.tot_operation + ' ~ ' +
            "Win Rate: %.6f" % ((self.profit_trades / self.tot_operation) * 100) + ' ~ ' +
            "W/L ratio: %.6f" % ((self.total_gain / self.profit_trades)
                                 / (abs(self.total_loss) / (self.tot_operation - self.profit_trades))) + ' ~ ' +
            "Sharpe Ratio: %.6f" % ((self._total_profit + 0.012) /
                                    np.std(list(map(lambda x: x * 100, self.equity_trading_system)))) + ' ~ ' +
            "MDD: %.6f" % max_dd(self.wallet_series)

        )

        window_ticks = np.arange(len(self._position_history))
        prices_test = self.prices[(self.window_size - 1):]

        ax1.plot(prices_test)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == 0:
                short_ticks.append(tick)
            elif self._position_history[i] == 1:
                long_ticks.append(tick)

        ax1.plot(short_ticks, prices_test[short_ticks], 'ro')
        ax1.plot(long_ticks, prices_test[long_ticks], 'go')
        ax1.title.set_text('Buy/Sell signals')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Price')

        benchmark, = ax2.plot(np.arange(len(self.equity_benchmark)),
                              list(map(lambda x: x * 100, self.equity_benchmark)))
        trading_system, = ax2.plot(np.arange(len(self.equity_trading_system)),
                                   list(map(lambda x: x * 100, self.equity_trading_system)))
        ax2.title.set_text('Equity Line (%)')
        ax2.legend([benchmark, trading_system], ["Benchmark", "Trading System"])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Equity')

        plt.show()
