from gym_anytrading.envs import Actions, Positions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def max_dd(wallet_series: pd.Series) -> float:
    """
    Compute Maximum Draw Down
    :param wallet_series: time-series of the wallet
    """
    cumulative_returns = pd.Series(wallet_series)
    highwatermarks = cumulative_returns.cummax()
    drawdowns = 1 - (1 + cumulative_returns) / (1 + highwatermarks)
    max_drawdown = max(drawdowns)

    return max_drawdown * 100


class Wallet(object):
    """
    It stores info for performances
    """
    def __init__(self, wallet, starting_price, compute_commissions):
        super().__init__()
        self.history = {"EquityTradingSystem": [],
                        "EquityBenchmark": [],
                        "ProfitLoss": [],
                        "WalletSeries": [],
                        "Position": []}

        self.starting_wallet = wallet
        self.wallet = self.starting_wallet
        self.cap_inv = 0.5 * self.starting_wallet
        self.starting_price_benchmark = starting_price
        self.total_reward = 0
        self.tot_operation = 0
        self.profit_trades = 0
        self.total_gain = 0
        self.total_loss = 0
        self.tot_commissions = 0
        self.compute_commissions = compute_commissions

    def step(self,
             action: tuple,
             price_enter: float,
             last_price: float,
             current_position: int,
             reward_step: float,
             shares_months: int):
        """
        Perform wallet step to update infos

        :param action: (action, action_prob)
        :param price_enter: enter price of the last transition Short.Position -> Buy
        :param last_price: the last price beaten
        :param current_position: the current position
        :param reward_step:
        :param shares_months: number of shares traded in the current month
        :return: info
        """
        equity_ts_step, equity_benchmark_step, pl_step, wallet_step, step_position = \
            self._compute_info(action[0], price_enter, last_price, current_position, reward_step, shares_months)

        shares_long = self._update_cap_inv(action, current_position, last_price)

        info = dict(
            EquityTradingSystem=equity_ts_step,
            EquityBenchmark=equity_benchmark_step,
            ProfitLoss=pl_step,
            WalletSeries=wallet_step,
            Position=step_position)

        self._update_history(info)

        return info, shares_long

    def _compute_info(self,
                      action: int,
                      price_enter: float,
                      last_price: float,
                      current_position: int,
                      step_reward: float,
                      shares_months: int):
        """
        Compute info to update the performances history

        :param action: (action, action_prob)
        :param price_enter: enter price of the last transition Short.Position -> Buy
        :param last_price: the last price beaten
        :param current_position: the current position
        :param step_reward:
        :param shares_months: number of shares traded in the current month
        :return: info
        """
        commission = 0
        # If a trading trajectory Buy/Sell is closed with the current action then compute commissions
        if action == Actions.Sell.value and current_position == Positions.Long:
            commission = self.compute_commissions(self.cap_inv, shares_months, last_price) * 2
            self.tot_commissions += commission
        # If there is an a long position or a long position is closed with the current action then update performances
        if (action == Actions.Sell.value and current_position == Positions.Long) or \
                (action == Actions.Buy.value and current_position == Positions.Long):
            # Update equity time-series step
            equity_ts_step = (self.total_reward + (last_price - price_enter)
                              / price_enter * self.cap_inv - commission) / self.starting_wallet
            # Update pl time-series step
            pl_step = (last_price - price_enter) / price_enter * self.cap_inv - commission
            # Update wallet time-series step
            wallet_step = self.wallet + (last_price - price_enter) / price_enter * self.cap_inv - commission
            # If a trading trajectory Buy/Sell is closed with the current action
            if action == Actions.Sell.value and current_position == Positions.Long:
                # Update total reward
                self.total_reward = wallet_step - self.starting_wallet
                # Update wallet
                self.wallet += step_reward
                # If a trading trajectory Buy/Sell is closed with the current action
                if step_reward > 0:
                    # Update the number of profit trades
                    self.profit_trades += 1
                    # Update the total gain realized
                    self.total_gain += step_reward
                else:
                    # Update the total loss realized
                    self.total_loss += step_reward
                # Update the number of trading trajectory completed
                self.tot_operation += 1
                # Save the position
                step_position = action
            else:
                # Position is set to None in order to don't display equal consecutive trading signal in the final plot
                step_position = None
        # If a trading trajectory Buy/Sell is not closed with the current action
        else:
            # Valid for the first step of the environment
            if len(self.history["EquityTradingSystem"]) == 0:
                equity_ts_step = 0
                step_position = 0
            else:
                # The equity step is equal to the last equity step
                equity_ts_step = self.history["EquityTradingSystem"][-1]
                if action == Actions.Sell.value and current_position == Positions.Short:
                    # Position = None in order to don't display equal consecutive trading signal in the final plot
                    step_position = None
                else:
                    step_position = action
            pl_step = 0
            wallet_step = self.wallet

        # Update equity benchmark
        equity_benchmark_step = (last_price - self.starting_price_benchmark) / self.starting_price_benchmark

        return equity_ts_step, equity_benchmark_step, pl_step, wallet_step, step_position

    def _update_cap_inv(self,
                        action: tuple,
                        current_position: int,
                        last_price: float) -> float:
        """
        Update cap_inv used to compute rewards and performances metrics computation

        :param action: (action, action_prob)
        :param last_price: the last price beaten
        :param current_position: the current position
        :return: number of traded shares
        """
        shares_long = 0
        if ((action[0] == Actions.Sell.value and current_position == Positions.Long)
                or (action[0] == Actions.Buy.value and current_position == Positions.Short)):
            self.cap_inv = math.exp((action[1][0][action[0]].item() - 1) / 0.34) * self.wallet
            # If you were Short and the chosen action is Buy
            if action[0] == Actions.Buy.value and current_position == Positions.Short:
                # Update number of shares
                shares_long = self.cap_inv / last_price
                # shares_long = 60000 / self.cap_inv
        return shares_long

    def _update_history(self,
                        info: dict) -> None:
        """
        Update history of performances

        :param info: dictionary of infos
        :return:
        """
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render_all(self,
                   prices: pd.DataFrame) -> None:
        """
        Plot performances

        :param prices: price time-series
        :return:
        """
        fig, (ax1, ax2) = plt.subplots(2)

        fig.suptitle(
            "Total Reward: %.2f" % self.total_reward + ' ~ ' +
            "Total Commission: %.2f" % self.tot_commissions + ' ~ ' +
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
        ax2.set_ylabel('wallet_step / wallet_0')

        plt.show()
