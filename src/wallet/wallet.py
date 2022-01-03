import numpy as np
import pandas as pd
from enum import Enum
import math

from gym_anytrading.envs import Actions


def max_dd(wallet_series: list) -> float:
    """
    Compute Maximum Draw Down
    :param wallet_series: time-series of the wallet
    """
    cumulative_returns = pd.Series(wallet_series)
    highwatermarks = cumulative_returns.cummax()
    drawdowns = 1 - (1 + cumulative_returns) / (1 + highwatermarks)

    if len(drawdowns) == 0:
        return 0

    max_drawdown = max(drawdowns)

    return max_drawdown * 100


class Wallet(object):
    """
    It stores info for performances
    """

    def __init__(self,
                 wallet,
                 bet_size_factor,
                 pip,
                 leverage,
                 wandb):
        """
        :param wallet: amount of starting wallet
        :param starting_price: the close price of the first observation (used to compute benchmark performances)
        :param bet_size_factor: factor to increase (reduce) the bet size
        :param compute_commissions: function for computing commissions
        :return
        """
        super().__init__()
        self.last_commissions_paid = 0
        self.history = {"EquityTradingSystem": [],
                        "ProfitLoss": [],
                        "WalletSeries": [],
                        "Position": [],
                        "PipPL": [],
                        "Commissions": []}

        self.wandb = wandb
        self.starting_wallet = wallet
        self.pip = pip
        self.wallet = self.starting_wallet
        self.cap_inv = 0.0
        self.bet_size_factor = bet_size_factor
        self.total_reward = 0
        self.tot_operation = 0
        self.profit_trades = 0
        self.total_gain = 0
        self.total_loss = 0
        self.tot_commissions = 0
        self.bet_size = 0
        self.tot_pip_pl = 0
        self.leverage = leverage

    def step(self,
             action: tuple,
             price_enter: float,
             last_price: float,
             current_position: int):
        """
        Perform wallet step to update info

        :param action: (action, action_prob)
        :param price_enter: enter price of the last transition Short.Position -> Buy
        :param last_price: the last price beaten
        :param current_position: the current position
        :param reward_step:
        :param shares_months: number of shares traded in the current month
        :return: info
        """
        equity_ts_step, pl_step, wallet_step, commission, pip_pl = \
            self._compute_info(action[0], price_enter, last_price, current_position)

        self.tot_commissions += commission
        self.last_commission_paid = commission
        self.tot_pip_pl += pip_pl * 10000

        self._update_cap_inv(action)

        info = dict(
            EquityTradingSystem=equity_ts_step,
            ProfitLoss=pl_step,
            WalletSeries=wallet_step,
            PipPL=pip_pl * 10000,
            Commissions=commission)

        self._update_history(info)

        self.std_deviation = np.std(self.history["EquityTradingSystem"])

        if self.std_deviation == 0:
            self.sharpe_ratio = 0.00
        else:
            self.sharpe_ratio = ((self.wallet - self.starting_wallet) / self.starting_wallet) \
                                / self.std_deviation

        self.mdd = max_dd(self.history["WalletSeries"])

        if self.mdd == 0:
            self.romad = 0.00
        else:
            self.romad = ((self.wallet - self.starting_wallet) / self.starting_wallet) / self.mdd * 100

        self.wandb.log({"metrics/equity": equity_ts_step,
                        "metrics/std_deviation": self.std_deviation,
                        "metrics/sharpe_ratio": self.sharpe_ratio,
                        "metrics/mdd": self.mdd,
                        "metrics/romad": self.romad,
                        "metrics/pip_pl": self.tot_pip_pl})

        return info

    def _compute_info(self,
                      action: int,
                      price_enter: float,
                      last_price: float,
                      current_position: int):
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

        commissions = 0
        pip_pl = 0
        equity_ts_step = 0
        pl_step = 0
        wallet_step = self.wallet
        self.last_position = current_position

        if action == 1 and current_position == 1:
            pl_step = (last_price - self.pip - price_enter + self.pip) * self.cap_inv
            equity_ts_step = (self.total_reward + pl_step) / self.starting_wallet
            wallet_step = self.wallet + pl_step

        if action == 0 and current_position == 0:
            pl_step = (price_enter - self.pip - last_price + self.pip) * self.cap_inv
            equity_ts_step = (self.total_reward + pl_step) / self.starting_wallet
            wallet_step = self.wallet + pl_step

        if action == 0 and current_position == 1:
            pip_pl = last_price - self.pip - price_enter + self.pip
            pl_step = pip_pl * self.cap_inv
            commissions = self.pip * 2 * self.cap_inv
            equity_ts_step = (self.total_reward + pl_step) / self.starting_wallet
            wallet_step = self.wallet + pl_step
            self.total_reward = wallet_step - self.starting_wallet
            self.wallet = wallet_step

            if pl_step > 0:
                # Update the number of profit trades
                self.profit_trades += 1
                # Update the total gain realized
                self.total_gain += pl_step
            else:
                # Update the total loss realized
                self.total_loss += pl_step
            # Update the number of trading trajectory completed
            self.tot_operation += 1

        if action == 1 and current_position == 0:
            pip_pl = price_enter - self.pip - last_price + self.pip
            pl_step = pip_pl * self.cap_inv
            commissions = self.pip * 2 * self.cap_inv
            equity_ts_step = (self.total_reward + pl_step) / self.starting_wallet
            wallet_step = self.wallet + pl_step
            self.total_reward = wallet_step - self.starting_wallet
            self.wallet = wallet_step

            if pl_step > 0:
                # Update the number of profit trades
                self.profit_trades += 1
                # Update the total gain realized
                self.total_gain += pl_step
            else:
                # Update the total loss realized
                self.total_loss += pl_step
            # Update the number of trading trajectory completed
            self.tot_operation += 1

        # Update equity benchmark
        return equity_ts_step, pl_step, wallet_step, commissions, pip_pl

    def _update_cap_inv(self,
                        action: tuple):
        """
        Update cap_inv used to compute rewards and performances metrics computation

        :param action: (action, action_prob)
        """
        if (action[0] == 1 and self.last_position == 0) or \
                (action[0] == 0 and self.last_position == 1):
            # self.cap_inv = math.exp((action[1][0][action[0]].item() - 1) / self.bet_size_factor) * self.wallet
            self.bet_size = math.exp((action[1][0][action[0]].item() - 1) / self.bet_size_factor)
            if self.leverage:
                leverage = int(30 * self.bet_size)
            else:
                leverage = 1
            self.cap_inv = self.bet_size * self.wallet * leverage

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

    def wandb_final(self) -> None:
        """
        Plot performances

        :param prices: price time-series
        :return:
        """
        std_deviation = np.std(self.history["EquityTradingSystem"])
        total_percent_profit = (self.wallet - self.starting_wallet) / self.starting_wallet * 100
        win_rate = (self.profit_trades / self.tot_operation) * 100
        w_l_ratio = (self.total_gain / self.profit_trades) \
                    / (abs(self.total_loss) / (self.tot_operation - self.profit_trades))
        sharpe_ratio = ((self.wallet - self.starting_wallet) / self.starting_wallet) \
                       / std_deviation
        mdd = max_dd(self.history["WalletSeries"])
        romad = total_percent_profit / mdd

        self.wandb.run.summary["Total_Percent_Profit"] = total_percent_profit
        self.wandb.run.summary["Win_Rate"] = win_rate
        self.wandb.run.summary["Win_Loss_Ratio"] = w_l_ratio
        self.wandb.run.summary["Sharpe_Ratio"] = sharpe_ratio
        self.wandb.run.summary["Maximum_Drawdown"] = mdd
        self.wandb.run.summary["Romad"] = romad
        self.wandb.run.summary["Std_deviation"] = std_deviation
        self.wandb.run.summary["Commissions"] = sum(self.history['Commissions'])
        self.wandb.run.summary["Profit/Loss"] = self.wallet - self.starting_wallet
        self.wandb.run.summary["Total_Number_Trades"] = self.tot_operation
