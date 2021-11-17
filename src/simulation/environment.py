from gym_anytrading.envs import StocksEnv, Actions, Positions
import pandas as pd
from hurst import compute_Hc
from src.wallet.wallet import Wallet
import torch


def my_process_data(env):
    """
    Extract prices and features for each environment step

    :param env:
    :return:
    """
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['time', 'open', 'high', 'low', 'volume', 'close',
                                     'WeekDay', 'MonthDay', 'Month', 'Hour', 'Minute']].to_numpy()[start:end]
    return prices, signal_features


def compute_commissions(cap_inv: float,
                        shares_months: float,
                        last_price: float) -> float:
    """
    Simulation of the IB's degressive plan commission (https://www.interactivebrokers.eu/it/index.php?f=5688&p=stocks2)

    :param cap_inv: invested capital
    :param shares_months: number of shares traded in the current month
    :param last_price: last price beaten

    :return: commissions paid
    """

    commissions = 0
    shares = cap_inv / last_price

    if shares_months < 300000:
        commissions = max(0.35, shares * 0.0035)
        commissions = min(commissions, 0.01 * cap_inv)
    if 300000 < shares_months <= 3000000:
        commissions = max(0.35, shares * 0.002)
        commissions = min(commissions, 0.01 * cap_inv)
    if 3000000 < shares_months <= 20000000:
        commissions = max(0.35, shares * 0.0015)
        commissions = min(commissions, 0.01 * cap_inv)
    if 20000000 < shares_months <= 100000000:
        commissions = max(0.35, shares * 0.001)
        commissions = min(commissions, 0.01 * cap_inv)
    if shares_months > 100000000:
        commissions = max(0.35, shares * 0.0005)
        commissions = min(commissions, 0.01 * cap_inv)
    """
    # Binance commissions
    if shares_months < 4500:
        commissions = 0.00075 * cap_inv
    if 4500 <= shares_months < 10000:
        commissions = 0.000675 * cap_inv
    if 10000 <= shares_months < 20000:
        commissions = 0.0006 * cap_inv
    if 20000 <= shares_months < 40000:
        commissions = 0.000525 * cap_inv
    if 40000 <= shares_months < 80000:
        commissions = 0.000450 * cap_inv
    if 80000 <= shares_months < 150000:
        commissions = 0.000375 * cap_inv
    if shares_months > 150000:
        commissions = 0.0003 * cap_inv
    """

    return commissions


class Environment(StocksEnv):
    """
    Trading environment
    """
    _process_data = my_process_data

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int,
                 frame_bound: tuple,
                 starting_wallet: float,
                 bet_size_factor: float):
        """
        :param df: dataframe used to compile the trading simulation
        :param window_size: Number of ticks (current and previous ticks) returned as a Gym observation.
        :param frame_bound: A tuple which specifies the start and end of df
        :param starting_wallet: the initial capital available to the agent
        :return:
        """
        super().__init__(df, window_size, frame_bound)
        self._start_tick = self.window_size
        self._end_tick = len(self.prices)
        self.counter = self.window_size - 1
        self.wallet = Wallet(starting_wallet,
                             self.prices[self._start_tick - 1],
                             bet_size_factor,
                             compute_commissions)
        self.shares_months = 0
        self.month = 0
        self.last_price_short = 0
        self.last_price_long = 0

    def reset(self) -> tuple:
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick
        self._position = Positions.Short
        obs = self._get_observation()
        self.last_price_short = obs[-2, 5]
        info = self.return_info(obs)
        self.month = int(obs[-2, 0][5:7])
        # return the dataframe except the first column because it encodes the date
        return obs[:-1, 1:6], info

    def _calculate_reward(self,
                          action: tuple) -> tuple:
        """
        Compute the rewards given the chosen action
        :param action: chosen action
        :return: (reward, done)
        """
        done = False
        price_1 = price_2 = denominator = 0
        commission = compute_commissions(self.wallet.cap_inv,
                                         self.shares_months,
                                         self.prices[self._current_tick - 2]) * 2
        """
        Case 1:
        if I held a Long position open and decided to sell -> the reward is the profit obtained, namely the percentage
        increase (decrease) of the price multiplied by the invested capital minus the commissions; done is set to True 
        because a trajectory buy/sell is completed 
        """
        if action[0] == Actions.Sell.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._last_trade_tick]
            denominator = price_2
            done = True
        """
        Case 2:
        if I held a Short position open and decided to buy -> the reward is the profit obtained as if you had made 
        a short sell, namely (minus) the percentage increase (decrease) of the price multiplied by the invested capital; 
        done is set to True because a trajectory sell/buy is completed 
        """
        if action[0] == Actions.Buy.value and self._position == Positions.Short:
            price_1 = self.prices[self._last_trade_tick]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_1
            commission = -commission
            done = True
        """
        Case 3:
        if I held a Long position open and decided to buy -> the reward is the profit obtained as if you had opened 
        the long position in the previous time-frame and you had sold in the current time-frame, 
        namely the percentage increase (decrease) of the price multiplied by the invested capital; 
        done is False in this case
        """
        if action[0] == Actions.Buy.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 1]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_2
            commission = -commission
        """
        Case 4:
        if I held a Short position open and decided to sell -> the reward is the profit obtained as if you had opened 
        the short position in the previous time-frame and you had bought in the current time-frame, 
        namely (minus) the percentage increase (decrease) of the price multiplied by the invested capital; 
        done is False in this case
        """
        if action[0] == Actions.Sell.value and self._position == Positions.Short:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._current_tick - 1]
            denominator = price_1
            commission = -commission

        step_reward = (price_1 - price_2) / denominator * self.wallet.cap_inv - commission

        return step_reward, done

    def step(self,
             action: tuple) -> tuple:
        """
        Perform a step environment
        :param action: (action, prob_action)
        :return: tuple containing step information
        """
        self._done = False

        # Perform step environment
        step_reward, done_trajectory = self._calculate_reward(action)

        # Perform wallet step to update metric performances
        info_wallet, shares_long = self.wallet.step(action,
                                                    self.prices[self._last_trade_tick],
                                                    self.prices[self._current_tick - 2],
                                                    self._position,
                                                    step_reward,
                                                    self.shares_months)

        # Update last trade tick index and flip current position when the position is flipped due to the chosen action
        self.update_last_trade_tick_and_position(action[0])

        # Slice the time-window
        self._current_tick += 1
        observation = self._get_observation()
        info = self.return_info(observation)
        # Reset the shares traded if the month is changed
        self.update_shares_months(observation, shares_long)
        # Stop the simulation if you are at the end of the dataframe or if your wallet is empty
        if self._current_tick == self._end_tick or self.wallet.wallet <= 0:
            self._done = True

        self.counter += 1

        return (observation[:-1, 1:6], info), step_reward, self._done, info_wallet, done_trajectory

    def get_position(self):
        """
        Return the current position
        """
        return self._position

    def update_shares_months(self,
                             observation: pd.Series,
                             shares_long: float) -> None:
        """
        Reset the counter for the number of shares traded in the current month
        :param observation: last observation
        :param shares_long: the number of shares traded in the current step
        :return: tuple containing step information
        """
        # If the last observation is the first observation of a new month
        if int(observation[-1, 0][5:7]) != self.month:
            # Update the current month
            self.month = int(observation[-1, 0][5:7])
            # Reset the counter for the number of the traded shares
            self.shares_months = 0
        else:
            # Update the counter for the number of the traded shares
            self.shares_months += shares_long * 2

    def update_last_trade_tick_and_position(self,
                                            action: int) -> None:
        """
        Update last trade tick and position
        :param action: chosen action
        :return:
        """
        if action == Actions.Buy.value and self._position == Positions.Short or \
                action == Actions.Sell.value and self._position == Positions.Long:
            self._last_trade_tick = self._current_tick - 2
            self._position = self._position.opposite()

            if action == Actions.Buy.value:
                self.last_price_long = self.prices[self._last_trade_tick]
            else:
                self.last_price_short = self.prices[self._last_trade_tick]

    def return_info(self,
                    observation: pd.Series) -> torch.Tensor:
        """
        Build info tensor: [profit/loss of the current trade, hurst, number of shares traded in the current month,
        current position, week day, month day, month, hour, minute]

        :param observation: info are extracted by considering the current observation
        :return: tensor info
        """
        # Profit/loss computation depends from the current position (long/short trade or short selling)
        if self._position == 0:
            p_l = (observation[-2, 5] - self.last_price_long) / self.last_price_long
        else:
            p_l = (self.last_price_short - observation[-2, 5]) / self.last_price_short

        # Compute Hurst exponent
        hurst = compute_Hc(observation[:-1, 5], kind='price', simplified=True)[0]

        return torch.tensor([p_l,
                             hurst,
                             self.shares_months / 100000000,
                             self._position.value,
                             observation[-1, 6],
                             observation[-1, 7],
                             observation[-1, 8],
                             observation[-1, 9],
                             observation[-1, 10]],
                            dtype=torch.float32).unsqueeze(dim=0)

    def render_performances(self):
        """
        Render performances
        """
        self.wallet.render_all(self.prices[(self.window_size - 1):-1])
