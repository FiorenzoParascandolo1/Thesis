from gym_anytrading.envs import StocksEnv, Actions, Positions
import pandas as pd
from hurst import compute_Hc
import math
import numpy as np
from src.wallet.wallet import Wallet
import torch
from src.data_utils.preprocessing_utils import StackImages, GADFTransformation, ManageSymmetries, \
    IdentityTransformation, ManagePeriods
from torchvision.transforms import transforms
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import io
import plotly.io as pio
from PIL import Image
import matplotlib.pyplot as plt

pio.renderers.default = 'browser'


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

    return commissions


def return_action_name(action: int,
                       position: int) -> str:
    if position == 1 and action == 1:
        return "HOLD"
    elif position == 1 and action == 0:
        return "SELL"
    elif position == 0 and action == 1:
        return "BUY"
    elif position == 0 and action == 0:
        return "DO NOTHING"


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
                 bet_size_factor: float,
                 periods: list,
                 pixels: int,
                 manage_symmetries: bool,
                 render: bool,
                 name: str,
                 wandb):
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
        self.wallet = Wallet(starting_wallet,
                             self.prices[self._start_tick - 1],
                             bet_size_factor,
                             wandb,
                             compute_commissions)
        self.periods = periods
        self.render = render

        if self.render:
            # Asset name
            self.name = name
            # Granularity of subplots
            self.subplots_titles = [f"{self.periods[i] * 5}-min granularity" for i in range(len(self.periods))]
            # Observations candlestick plot
            self.olhc_1 = make_subplots(rows=int(len(self.periods) / 2), cols=2, subplot_titles=self.subplots_titles,
                                        horizontal_spacing=0.05, vertical_spacing=0.09)
            # Explanation candlestick plot
            self.olhc_2 = make_subplots(rows=int(len(self.periods) / 2), cols=2, subplot_titles=self.subplots_titles,
                                        horizontal_spacing=0.05, vertical_spacing=0.09)

            # Figure and axes dimension settings
            self.fig = plt.figure(constrained_layout=False, figsize=(50, 50))
            self.gs = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 15])
            self.ax1 = self.fig.add_subplot(self.gs[1, 0])
            self.ax2 = self.fig.add_subplot(self.gs[1, 1])
            self.ax3 = self.fig.add_subplot(self.gs[0, :])

            # Titles of the performance indices of the table
            self.raw_table_performances = ["Profit/Loss",
                                           "Commissions",
                                           "Gross Assets",
                                           "Liquid Assets",
                                           "Invested Assets",
                                           "Tot. Profit/Loss",
                                           "Tot. Profit/Loss (%)",
                                           "Commissions Paid",
                                           "Std",
                                           "Sharpe Ratio",
                                           "MDD (%)",
                                           "RoMaD",
                                           "Total Trades",
                                           "Profit Trades",
                                           "Loss Trades",
                                           "Win Rate (%)"]

        self.manage_periods = ManagePeriods(pixels=pixels, periods=periods)
        self.transform = transforms.Compose([GADFTransformation(periods=periods,
                                                                pixels=pixels),
                                             ManageSymmetries(pixels=pixels) if manage_symmetries
                                             else IdentityTransformation(),
                                             StackImages()])
        self.shares_months = 0
        self.month = 0
        self.last_price_short = 0
        self.last_price_long = 0
        self.last_obs = None
        self.pixels = pixels

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
        aggregated_series = self.manage_periods(obs[:-1, 0:6])
        self.last_obs = aggregated_series
        return self.transform(aggregated_series), info

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
        position = 0 if self.get_position() in [Positions.Short] else 1

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

        if self.render:
            self.render_environment(action, position)

        # Slice the time-window
        self._current_tick += 1
        observation = self._get_observation()
        info = self.return_info(observation)
        new_position = 0 if self.get_position() in [Positions.Short] else 1

        # Reset the shares traded if the month is changed
        self.update_shares_months(observation, shares_long)
        # Stop the simulation if you are at the end of the dataframe or if your wallet is empty
        if self._current_tick == self._end_tick or self.wallet.wallet <= 0:
            self._done = True

        reward_trajectory = self.compute_reward_end_trajectory_new(position, new_position, observation)

        aggregated_series = self.manage_periods(observation[:-1, 0:6])
        self.last_obs = aggregated_series

        return (self.transform(aggregated_series), info), \
               step_reward, self._done, info_wallet, done_trajectory, reward_trajectory

    def get_position(self):
        """
        Return the current position
        """
        return self._position

    def compute_reward_end_trajectory_new(self,
                                          position: int,
                                          new_position: int,
                                          observation: pd.Series) -> float:

        reward_trajectory = None

        if new_position == 1 and position == 0:
            reward_trajectory = (observation[-2, 6] - observation[-3, 6]) / observation[-3, 6] * self.wallet.cap_inv
        if new_position == 0 and position == 1:
            reward_trajectory = (observation[-3, 6] - observation[-2, 6]) / observation[-3, 6] * self.wallet.cap_inv

        return reward_trajectory

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
        hurst = compute_Hc(observation[-101:-1, 5], kind='price', simplified=True)[0]

        if self._position.value == 0:
            short = 1.0
            long = 0.0
        else:
            short = 0.0
            long = 1.0

        return torch.tensor([p_l,
                             hurst,
                             self.shares_months / 100000000,
                             long,
                             short,
                             observation[-2, 6],
                             observation[-2, 7],
                             observation[-2, 8],
                             observation[-2, 9],
                             observation[-2, 10]],
                            dtype=torch.float32).unsqueeze(dim=0)

    def render_performances(self):
        """
        Render performances
        """
        self.wallet.render_all(self.prices[(self.window_size - 1):-1])

    def render_environment(self,
                           action: tuple,
                           position: int) -> None:

        # Update Observation and Explanation plots for the current step
        self.update_olhc_graphs()

        # Observation axis computation
        buf = io.BytesIO()
        pio.write_image(self.olhc_1, buf, format='jpg', scale=3)
        img = Image.open(buf)
        self.ax1.axis('off')
        self.ax1.imshow(img)

        # Explanation axis computation
        buf = io.BytesIO()
        pio.write_image(self.olhc_2, buf, format='jpg', scale=3)
        img = Image.open(buf)
        self.ax2.imshow(img)
        self.ax2.axis('off')

        # Table axis computation
        self.ax3.clear()
        self.ax3.axis('off')
        table_values, colours = self.compute_table_values()
        the_table = self.ax3.table(cellText=table_values,
                                   colLabels=self.raw_table_performances,
                                   loc='center',
                                   cellLoc='center',
                                   colWidths=[0.061] * len(self.raw_table_performances),
                                   cellColours=colours)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1.2, 1.2)

        # Title computation
        string_title = return_action_name(action[0], position)
        self.fig.suptitle(self.name + " ~ " + str(self.last_obs[0].index[-1]) + " ~ " + "Action: " + string_title +
                          " (p = " + str(round(action[1][0][action[0]].item(), 2)) + ")")

        # Show the current step
        plt.show(block=False)
        plt.pause(5)

        # Reset Observation and Explanation plots
        self.olhc_1 = make_subplots(rows=int(len(self.periods) / 2), cols=2, subplot_titles=self.subplots_titles,
                                    horizontal_spacing=0.05, vertical_spacing=0.05)
        self.olhc_2 = make_subplots(rows=int(len(self.periods) / 2), cols=2, subplot_titles=self.subplots_titles,
                                    horizontal_spacing=0.05, vertical_spacing=0.05)

    def compute_table_values(self):

        # Compute table_values and relative colours for the current step
        table_values = [[str(round(self.wallet.history["ProfitLoss"][-1], 2)) + "$",
                         str(round(-self.wallet.last_commissions_paid, 2)) + "$",
                         str(round(self.wallet.history["WalletSeries"][-1], 2)) + "$",
                         str(round(self.wallet.wallet - self.wallet.cap_inv, 2)) + "$"
                         if self.wallet.last_position == 1 else str(round(self.wallet.wallet, 2)) + "$",
                         str(round(self.wallet.cap_inv, 2)) + "$" if self.wallet.last_position == 1 else str(
                             0.00) + "$",
                         str(round(self.wallet.history["WalletSeries"][-1] -
                                   self.wallet.starting_wallet, 2)) + "$",
                         str(round((self.wallet.history["WalletSeries"][-1] -
                                    self.wallet.starting_wallet) / self.wallet.starting_wallet,
                                   5)),
                         str(round(-self.wallet.tot_commissions, 2)) + "$",
                         str(round(self.wallet.std_deviation, 5)),
                         str(round(self.wallet.sharpe_ratio, 2)),
                         str(-round(self.wallet.mdd, 2)),
                         str(round(self.wallet.romad, 2)),
                         str(self.wallet.tot_operation),
                         str(self.wallet.profit_trades),
                         str(self.wallet.tot_operation - self.wallet.profit_trades),
                         str(round(self.wallet.profit_trades / self.wallet.tot_operation if
                                   self.wallet.tot_operation != 0 else 0.00, 2) * 100)]]
        colours = [['w' if int(self.wallet.history["ProfitLoss"][-1]) == 0
                    else 'g' if self.wallet.history["ProfitLoss"][-1] > 0 else 'r',
                    'w' if int(-self.wallet.last_commissions_paid) == 0 else 'r',
                    'w' if int(self.wallet.history["ProfitLoss"][-1]) == 0
                    else 'g' if self.wallet.history["ProfitLoss"][-1] > 0 else 'r',
                    'w' if int(self.wallet.history["ProfitLoss"][-1]) == 0
                    else 'g' if self.wallet.history["ProfitLoss"][-1] > 0 else 'r',
                    'w',
                    'w' if int(self.wallet.history["WalletSeries"][-1] -
                               self.wallet.starting_wallet) == 0
                    else 'g' if self.wallet.history["WalletSeries"][-1] -
                                self.wallet.starting_wallet > 0 else 'r',
                    'w' if int(self.wallet.history["WalletSeries"][-1] -
                               self.wallet.starting_wallet) == 0
                    else 'g' if self.wallet.history["WalletSeries"][-1] -
                                self.wallet.starting_wallet > 0 else 'r',
                    'r',
                    'w',
                    'w' if int(self.wallet.sharpe_ratio) == 0
                    else 'g' if self.wallet.sharpe_ratio > 0 else 'r',
                    'r',
                    'w' if int(self.wallet.romad) == 0
                    else 'g' if self.wallet.romad > 0 else 'r',
                    'w',
                    'g',
                    'r',
                    'w']]

        return table_values, colours

    def update_olhc_graphs(self):
        # Add to each cycle the subplot relating to the corresponding granularity of the observation
        for i in range(1, len(self.periods) + 1):
            # Compute the index (in plotly indices start from 1)
            row = int(math.ceil(float(i) / 2))
            if i % 2 == 0:
                col = 2
            else:
                col = 1

            # Extract hour:minute from DateTime index
            x = list(map(lambda x: str(x)[11:16], self.last_obs[i - 1].index.tolist()))

            # Update the observation plot
            self.olhc_1.add_trace(go.Candlestick(x=x,
                                                 open=self.last_obs[i - 1]['Open'].tolist(),
                                                 high=self.last_obs[i - 1]['High'].tolist(),
                                                 low=self.last_obs[i - 1]['Low'].tolist(),
                                                 close=self.last_obs[i - 1]['Close'].tolist()),
                                  row=row, col=col)

            # Update the explanation plot. TODO: adjust for explanation
            self.olhc_2.add_trace(go.Candlestick(x=x,
                                                 open=self.last_obs[i - 1]['Open'].tolist(),
                                                 high=self.last_obs[i - 1]['High'].tolist(),
                                                 low=self.last_obs[i - 1]['Low'].tolist(),
                                                 close=self.last_obs[i - 1]['Close'].tolist()),
                                  row=row, col=col)

            # the BUY / SELL labels are added to the graph of the observation with minimum granularity
            if i == 1:
                # Get the history of the positions in the market (pixel = number of observations for each granularity)
                positions = self.wallet.history["Position"][-self.pixels:]
                # When there are less than pixels positions, positions list is filled with None values
                if len(positions) < self.pixels:
                    positions = positions[0:]
                    fill_positions = [None for _ in range(self.pixels - len(positions))]
                    positions = fill_positions + positions

                # Compute marker position for labels BUY/SELL
                markers = [self.last_obs[i - 1]['High'].tolist()[j] +
                           0.15 if self.last_obs[i - 1]['Close'].tolist()[j] > self.last_obs[i - 1]['Open'].tolist()[j]
                           else self.last_obs[i - 1]['Low'].tolist()[j] - 0.15 for j in range(len(positions))]
                # Set to np.nan the markers for actions that not change the position on the market
                markers = [markers[j] if positions[j] is not None else np.nan for j in range(len(positions))]
                # Set text label for not None markers
                text = ["BUY" if positions[i] == 1 else "SELL" if positions[i] == 0 else np.nan
                        for i in range(len(positions))]
                # Add text labels to the observation plot
                self.olhc_1.add_trace(go.Scatter(x=x,
                                                 y=markers,
                                                 mode='text',
                                                 name='markers',
                                                 text=text,
                                                 textfont=dict(size=16, color="Black")
                                                 ), row=row, col=col)

        # Update layout (delete legend, set title and its font size)
        self.olhc_1.update_layout(xaxis1=dict(rangeslider=dict(visible=False)),
                                  xaxis2=dict(rangeslider=dict(visible=False)),
                                  xaxis3=dict(rangeslider=dict(visible=False)),
                                  xaxis4=dict(rangeslider=dict(visible=False)),
                                  xaxis5=dict(rangeslider=dict(visible=False)),
                                  xaxis6=dict(rangeslider=dict(visible=False)),
                                  xaxis7=dict(rangeslider=dict(visible=False)),
                                  xaxis8=dict(rangeslider=dict(visible=False)),
                                  margin=dict(b=0, l=0, r=0, t=50),
                                  title={'text': "Observation",
                                         'y': 0.99,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         'font': dict(size=25)},
                                  autosize=False,
                                  showlegend=False,
                                  width=1500,
                                  height=1500)

        self.olhc_2.update_layout(xaxis1=dict(rangeslider=dict(visible=False)),
                                  xaxis2=dict(rangeslider=dict(visible=False)),
                                  xaxis3=dict(rangeslider=dict(visible=False)),
                                  xaxis4=dict(rangeslider=dict(visible=False)),
                                  xaxis5=dict(rangeslider=dict(visible=False)),
                                  xaxis6=dict(rangeslider=dict(visible=False)),
                                  xaxis7=dict(rangeslider=dict(visible=False)),
                                  xaxis8=dict(rangeslider=dict(visible=False)),
                                  margin=dict(b=0, l=0, r=0, t=50),
                                  title={'text': "Explanation",
                                         'y': 0.99,
                                         'x': 0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',
                                         'font': dict(size=25)},
                                  autosize=False,
                                  showlegend=False,
                                  width=1500,
                                  height=1500)
