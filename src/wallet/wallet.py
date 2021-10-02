from gym_anytrading.envs import Actions, Positions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def max_dd(wallet_series):
    cumulative_returns = pd.Series(wallet_series)
    highwatermarks = cumulative_returns.cummax()
    drawdowns = 1 - (1 + cumulative_returns) / (1 + highwatermarks)
    max_drawdown = max(drawdowns)

    return max_drawdown * 100


class Wallet(object):

    def __init__(self, wallet, starting_price):
        super().__init__()
        self.history = {"EquityTradingSystem": [],
                        "EquityBenchmark": [],
                        "ProfitLoss": [],
                        "WalletSeries": [],
                        "Position": [],
                        "Done": []}
        self.starting_wallet = wallet
        self.wallet = self.starting_wallet
        self.cap_inv = 0.5 * self.starting_wallet
        self.starting_price_benchmark = starting_price
        self.total_reward = 0
        self.tot_operation = 0
        self.profit_trades = 0
        self.total_gain = 0
        self.total_loss = 0

    def step(self, action, price_enter, last_price, current_position, reward_step):
        equity_ts_step, equity_benchmark_step, pl_step, wallet_step, step_position = \
            self._compute_info(action[0], price_enter, last_price, current_position, reward_step)

        done = self._update_cap_inv(action, current_position)

        info = dict(
            EquityTradingSystem=equity_ts_step,
            EquityBenchmark=equity_benchmark_step,
            ProfitLoss=pl_step,
            WalletSeries=wallet_step,
            Position=step_position,
            Done=done)

        self._update_history(info)

        return info

    def _compute_info(self, action, price_enter, last_price, current_position, step_reward):
        if (action == Actions.Sell.value and current_position == Positions.Long) or \
                (action == Actions.Buy.value and current_position == Positions.Long):
            equity_ts_step = (self.total_reward + (last_price - price_enter) / price_enter * self.cap_inv) \
                             / self.starting_wallet
            pl_step = (last_price - price_enter) / price_enter * self.cap_inv
            wallet_step = self.wallet + (last_price - price_enter) / price_enter * self.cap_inv
            if action == Actions.Sell.value and current_position == Positions.Long:
                self.total_reward = wallet_step - self.starting_wallet
                self.wallet += step_reward
                if step_reward > 0:
                    self.profit_trades += 1
                    self.total_gain += step_reward
                else:
                    self.total_loss += step_reward
                self.tot_operation += 1
                step_position = action
            else:
                step_position = None
        else:
            if len(self.history["EquityTradingSystem"]) == 0:
                equity_ts_step = 0
                step_position = 0
            else:
                equity_ts_step = self.history["EquityTradingSystem"][-1]
                if action == Actions.Sell.value and current_position == Positions.Short:
                    step_position = None
                else:
                    step_position = action
            pl_step = 0
            wallet_step = self.wallet

        equity_benchmark_step = (last_price - self.starting_price_benchmark) / self.starting_price_benchmark

        return equity_ts_step, equity_benchmark_step, pl_step, wallet_step, step_position

    def _update_cap_inv(self, action, current_position):
        done = False
        if ((action[0] == Actions.Sell.value and current_position == Positions.Long)
                or (action[0] == Actions.Buy.value and current_position == Positions.Short)):
            self.cap_inv = action[1][0][action[0]].item() * self.wallet
            done = True

        return done

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self, prices):
        fig, (ax1, ax2) = plt.subplots(2)

        print(self.wallet)
        fig.suptitle(
            "Total Reward: %.2f" % self.total_reward + ' ~ ' +
            "Total Percent Profit: %.2f" % ((self.wallet - self.starting_wallet) / self.starting_wallet * 100) + ' ~ ' +
            "Number of trades: %d" % self.tot_operation + ' ~ ' +
            "Win Rate: %.2f" % ((self.profit_trades / self.tot_operation) * 100) + ' ~ ' +
            "W/L ratio: %.2f" % ((self.total_gain / self.profit_trades)
                                 / (abs(self.total_loss) / (self.tot_operation - self.profit_trades))) + ' ~ ' +
            "Sharpe Ratio: %.2f" % (((self.wallet - self.starting_wallet) / self.starting_wallet + 0.00012) /
                                    np.std(self.history["EquityTradingSystem"])) + ' ~ ' +
            "MDD: %.2f" % max_dd(self.history["WalletSeries"])

        )

        window_ticks = np.arange(len(self.history["Position"]))
        prices_test = prices

        ax1.plot(prices_test)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self.history["Position"][i] == 0:
                short_ticks.append(tick)
            elif self.history["Position"][i] == 1:
                long_ticks.append(tick)

        ax1.plot(short_ticks, prices_test[short_ticks], 'ro')
        ax1.plot(long_ticks, prices_test[long_ticks], 'go')
        ax1.title.set_text('Buy/Sell signals')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Price')

        benchmark, = ax2.plot(np.arange(len(self.history["EquityBenchmark"])), self.history["EquityBenchmark"])
        trading_system, = ax2.plot(np.arange(len(self.history["EquityTradingSystem"])),
                                   self.history["EquityTradingSystem"])
        ax2.title.set_text('Equity Line')
        ax2.legend([benchmark, trading_system], ["Benchmark", "Trading System"])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Equity (wallet_step / wallet_0)')

        plt.show()
