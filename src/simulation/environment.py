from gym_anytrading.envs import StocksEnv, Actions, Positions
import pandas as pd
from src.wallet.wallet import Wallet


def my_process_data(env):
    """
    Extract prices and features for each environment step

    :param env:
    :return:
    """
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['time', 'open', 'high', 'low', 'volume', 'close']].to_numpy()[start:end]
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
                 starting_wallet: float):
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
                             compute_commissions)
        self.shares_months = 0

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick
        self._position = Positions.Short
        obs = self._get_observation()
        self.month = int(obs[-1, 0][5:7])
        # return the dataframe except the first column because it encodes the date
        return obs[:, 1:]

    def _calculate_reward(self,
                          action: tuple) -> tuple:
        """
        Compute the rewards given the choosen action
        :param action: choosen action
        :return: (reward, done)
        """
        done = compute_done = False
        price_1 = price_2 = denominator = commission = 0

        """
        Case 1:
        if I held a Long position open and decided to sell -> the reward is the profit obtained, namely the percentage
        increase (deacrease) of the price multipled by the invested capital minus the commissions; done is set to True 
        because a trajectory buy/sell is completed 
        """
        if action[0] == Actions.Sell.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._last_trade_tick]
            denominator = price_2
            commission = compute_commissions(self.wallet.cap_inv,
                                             self.shares_months,
                                             self.prices[self._current_tick - 2]) * 2
            done = True
        """
        Case 2:
        if I held a Short position open and decided to buy -> the reward is the profit obtained as if you had made 
        a short sell, namely (minus) the percentage increase (deacrease) of the price multipled by the invested capital; 
        done is set to True because a trajectory sell/buy is completed 
        """
        if action[0] == Actions.Buy.value and self._position == Positions.Short:
            price_1 = self.prices[self._last_trade_tick]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_1
            commission = -compute_commissions(self.wallet.cap_inv,
                                              self.shares_months,
                                              self.prices[self._current_tick - 2]) * 2
            done = True
        """
        Case 3:
        if I held a Long position open and decided to buy -> the reward is the profit obtained as if you had opened 
        the long position in the previous time-frame and you had sold in the current time-frame, 
        namely the percentage increase (deacrease) of the price multipled by the invested capital; 
        done is False in this case
        """
        if action[0] == Actions.Buy.value and self._position == Positions.Long:
            price_1 = self.prices[self._current_tick - 1]
            price_2 = self.prices[self._current_tick - 2]
            denominator = price_2
        """
        Case 4:
        if I held a Short position open and decided to sell -> the reward is the profit obtained as if you had opened 
        the short position in the previous time-frame and you had bought in the current time-frame, 
        namely (minus) the percentage increase (deacrease) of the price multipled by the invested capital; 
        done is False in this case
        """
        if action[0] == Actions.Sell.value and self._position == Positions.Short:
            price_1 = self.prices[self._current_tick - 2]
            price_2 = self.prices[self._current_tick - 1]
            denominator = price_1

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
        step_reward, done = self._calculate_reward(action)

        # Perform wallet step to update metric performances
        info, shares_long = self.wallet.step(action, self.prices[self._last_trade_tick],
                                             self.prices[self._current_tick - 2],
                                             self._position, step_reward,
                                             self.shares_months)

        # Update last trade tick index when the position is flipped due to the choosen action
        if action[0] == Actions.Buy.value and self._position == Positions.Short or \
                action[0] == Actions.Sell.value and self._position == Positions.Long:
            self._last_trade_tick = self._current_tick - 2

        # Flip current position if the action is the opposite of the current position
        if ((action[0] == Actions.Buy.value and self._position == Positions.Short) or
                (action[0] == Actions.Sell.value and self._position == Positions.Long)):
            self._position = self._position.opposite()

        # Slice the time-window
        self._current_tick += 1
        observation = self._get_observation()

        # Reset the shares traded if the month is changed
        if int(observation[-1, 0][5:7]) != self.month:
            self.month = int(observation[-1, 0][5:7])
            self.shares_months = 0
        else:
            self.shares_months += shares_long * 2

        # Stop the simulation if you are at the end of the dataframe or if your wallet is empty
        if self._current_tick == self._end_tick or self.wallet.wallet <= 0:
            self._done = True

        self.counter += 1

        return observation[:, 1:], step_reward, self._done, info, done

    def get_position(self):
        """
        Return the current position
        """
        return self._position

    def render_performances(self):
        """
        Render performances
        """
        self.wallet.render_all(self.prices[(self.window_size - 1):-1])
